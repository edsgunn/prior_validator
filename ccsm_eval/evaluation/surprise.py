"""Surprise evaluator: computes per-token observation surprise via model forward passes.

This is the core of the Phase 1 evaluation. For each trajectory and character
prompt, we compute:

    S(x_{1:T}, c) = sum_{t} σ_t * (-log p_θ(x_t | c, x_{<t}))

where σ_t = 1 for observation tokens and 0 for action tokens.

We also compute:
    - Per-token surprise profile (for decomposed analysis)
    - Unigram log-probabilities (for residualisation)
    - Cumulative and normalised surprise scalars
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from ccsm_eval.evaluation.batching import EncodedBatch, build_batches
from ccsm_eval.evaluation.model_loader import LoadedModel
from ccsm_eval.trajectories.base import (
    CounterfactualEdit,
    CounterfactualSurpriseResult,
    SurpriseResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class SurpriseEvaluator:
    """Runs model forward passes and extracts per-token surprise.

    Args:
        loaded_model: A LoadedModel from model_loader.load_model().
        batch_size:   Sequences per GPU batch.
        max_length:   Truncation length in tokens.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        batch_size: int = 4,
        max_length: int = 4096,
    ):
        self.loaded_model = loaded_model
        self.batch_size = batch_size
        self.max_length = max_length
        self._unigram_cache: dict[int, float] = {}  # token_id -> unigram logprob

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trajectory: Trajectory,
        text: str,
        obs_mask_chars: list[bool],
        sem_types_chars: list[str],
        prompt_id: str,
    ) -> SurpriseResult:
        """Evaluate a single trajectory.

        Args:
            trajectory:       The Trajectory (for quality scores and metadata).
            text:             Full formatted text (prompt + game).
            obs_mask_chars:   Character-level observation mask.
            sem_types_chars:  Character-level semantic type map.
            prompt_id:        Identifier of the character prompt.

        Returns:
            SurpriseResult with per-token and aggregate surprise.
        """
        results = self.evaluate_batch(
            [(trajectory, text, obs_mask_chars, sem_types_chars, prompt_id)]
        )
        return results[0]

    def evaluate_batch(
        self,
        items: list[tuple[Trajectory, str, list[bool], list[str], str]],
        # (trajectory, text, obs_mask_chars, sem_types_chars, prompt_id)
    ) -> list[SurpriseResult]:
        """Evaluate a batch of trajectories.

        Returns one SurpriseResult per input item (same order).
        """
        flat_items = [
            (traj.trajectory_id, prompt_id, text, obs_mask, sem_types)
            for traj, text, obs_mask, sem_types, prompt_id in items
        ]
        traj_by_id = {traj.trajectory_id: traj for traj, *_ in items}

        results: list[SurpriseResult] = []

        if self.loaded_model.backend == "transformers":
            for batch in build_batches(
                flat_items,
                self.loaded_model.tokenizer,
                self.batch_size,
                self.max_length,
            ):
                batch_results = self._forward_transformers(batch, traj_by_id)
                results.extend(batch_results)
        else:
            raise NotImplementedError(
                "vLLM backend for per-token log-probs requires a custom implementation. "
                "Use the transformers backend for Phase 1."
            )

        return results

    def evaluate_counterfactual(
        self,
        edit: CounterfactualEdit,
        trajectory: Trajectory,
        character_prompt: str,
        prompt_id: str,
        formatter,
    ) -> CounterfactualSurpriseResult:
        """Measure surprise on the observation tokens following a counterfactual edit.

        Returns surprise on `original_tokens` and `replacement_tokens` separately.
        """
        import copy

        def _surprise_for_obs_tokens(context_text: str, obs_tokens_text: str) -> float:
            """Compute surprise over obs_tokens given a context prefix."""
            full_text = context_text + obs_tokens_text
            n_context = len(
                self.loaded_model.tokenizer(context_text, return_tensors="pt")["input_ids"][0]
            )
            tokenized = self.loaded_model.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = tokenized["input_ids"].to(self._device())
            with torch.no_grad():
                logits = self.loaded_model.model(input_ids).logits  # (1, T, V)

            log_probs = F.log_softmax(logits, dim=-1)
            total_surprise = 0.0
            n_obs = 0
            for t in range(n_context, input_ids.shape[1] - 1):
                token_id = input_ids[0, t + 1].item()
                lp = log_probs[0, t, token_id].item()
                total_surprise += -lp
                n_obs += 1
            return total_surprise if n_obs > 0 else 0.0

        # Build context text (trajectory up to the edit point)
        # Use the formatter to get a prefix representation
        context_traj = copy.deepcopy(trajectory)
        context_traj.tokens = context_traj.tokens[: edit.edit_position]
        ctx_text, _, _ = formatter.format(context_traj, character_prompt)

        original_text = " ".join(t.text for t in edit.original_tokens)
        replacement_text = " ".join(t.text for t in edit.replacement_tokens)

        s_original = _surprise_for_obs_tokens(ctx_text, original_text)
        s_replacement = _surprise_for_obs_tokens(ctx_text, replacement_text)

        return CounterfactualSurpriseResult(
            trajectory_id=edit.trajectory_id,
            edit_position=edit.edit_position,
            direction=edit.direction,
            quality_delta=edit.quality_delta,
            surprise_original=s_original,
            surprise_replacement=s_replacement,
            delta_surprise=s_replacement - s_original,
            prompt_id=prompt_id,
            model_id=self.loaded_model.model_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _forward_transformers(
        self,
        batch: EncodedBatch,
        traj_by_id: dict[str, Trajectory],
    ) -> list[SurpriseResult]:
        import torch

        device = self._device()
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        with torch.no_grad():
            outputs = self.loaded_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, V)

        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Unigram log-probs for residualisation
        # We approximate unigram as p(x_t | <empty>) — model's prediction at position 0
        unigram_lps = self._get_unigram_logprobs(input_ids, log_probs)

        results: list[SurpriseResult] = []

        for b_idx, (traj_id, prompt_id) in enumerate(
            zip(batch.trajectory_ids, batch.prompt_ids)
        ):
            traj = traj_by_id[traj_id]
            obs_mask = batch.observation_masks[b_idx]
            sem_types = batch.semantic_types[b_idx]

            per_tok_surprise: list[float] = []
            per_tok_sem_type: list[str] = []
            per_tok_unigram: list[float] = []

            seq_len = input_ids.shape[1]

            # The model predicts x_t+1 from position t.
            # log p(x_t | context) is at logits[t-1, x_t].
            # We start from position 1 (the first token that can be predicted).
            for t in range(1, seq_len):
                if attention_mask[b_idx, t].item() == 0:
                    continue  # padding token

                is_obs = obs_mask[t] if t < len(obs_mask) else False
                if not is_obs:
                    continue  # skip action tokens

                token_id = input_ids[b_idx, t].item()
                lp = log_probs[b_idx, t - 1, token_id].item()
                surprise = -lp

                sem_type = sem_types[t] if t < len(sem_types) else "unknown"
                unigram_lp = unigram_lps.get(token_id, -math.log(self._vocab_size()))

                per_tok_surprise.append(surprise)
                per_tok_sem_type.append(sem_type)
                per_tok_unigram.append(unigram_lp)

            n_obs = len(per_tok_surprise)
            cumulative = sum(per_tok_surprise)
            normalised = cumulative / n_obs if n_obs > 0 else 0.0

            results.append(
                SurpriseResult(
                    trajectory_id=traj_id,
                    prompt_id=prompt_id,
                    model_id=self.loaded_model.model_id,
                    per_token_surprise=per_tok_surprise,
                    per_token_semantic_type=per_tok_sem_type,
                    per_token_unigram_logprob=per_tok_unigram,
                    cumulative_surprise=cumulative,
                    normalised_surprise=normalised,
                    quality_scores=traj.quality_scores,
                    quality_level=traj.quality_level,
                    environment=traj.environment,
                )
            )

        return results

    def _get_unigram_logprobs(
        self, input_ids: torch.Tensor, log_probs: torch.Tensor
    ) -> dict[int, float]:
        """Approximate unigram log-probs using the model's prediction at position 0.

        This gives log p(x | <BOS>) which approximates the model's marginal
        distribution over tokens, useful for residualising out token frequency.
        """
        if not self._unigram_cache:
            # Use the first sequence's first-position predictions as unigram proxy
            first_pos_lps = log_probs[0, 0, :].cpu()  # (V,)
            for token_id in range(first_pos_lps.shape[0]):
                self._unigram_cache[token_id] = first_pos_lps[token_id].item()
        return self._unigram_cache

    def _device(self) -> str:
        return self.loaded_model.device if self.loaded_model.device != "auto" else "cuda"

    def _vocab_size(self) -> int:
        return self.loaded_model.tokenizer.vocab_size
