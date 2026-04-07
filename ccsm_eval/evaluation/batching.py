"""Batching utilities for efficient trajectory evaluation.

Groups trajectories into batches sized for the available GPU memory,
handling variable-length sequences with left-padding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


@dataclass
class EncodedBatch:
    """A batch of tokenised trajectories ready for a model forward pass."""

    input_ids: torch.Tensor           # (batch, seq_len)
    attention_mask: torch.Tensor      # (batch, seq_len)
    observation_masks: list[list[bool]]  # per-sequence, per-token
    semantic_types: list[list[str]]      # per-sequence, per-token
    trajectory_ids: list[str]
    prompt_ids: list[str]


def build_batches(
    items: list[tuple[str, str, str, list[bool], list[str]]],
    # (trajectory_id, prompt_id, full_text, obs_mask_chars, sem_types_chars)
    tokenizer,
    batch_size: int,
    max_length: int = 4096,
) -> Iterator[EncodedBatch]:
    """Tokenise trajectory texts and yield batches.

    For each text, the character-level observation mask (obs_mask_chars) is
    mapped to a token-level mask using the token's start character position.
    A token inherits is_observation=True if its first character is in an
    observation region.

    Args:
        items: List of (traj_id, prompt_id, text, char_obs_mask, char_sem_types).
        tokenizer: HuggingFace tokenizer.
        batch_size: Sequences per batch.
        max_length: Truncate to this many tokens.

    Yields:
        EncodedBatch objects.
    """
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        traj_ids = [c[0] for c in chunk]
        prompt_ids = [c[1] for c in chunk]
        texts = [c[2] for c in chunk]
        char_obs_masks = [c[3] for c in chunk]
        char_sem_types = [c[4] for c in chunk]

        encoding = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        offset_maps = encoding.pop("offset_mapping")  # (batch, seq_len, 2)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Map character-level masks to token-level masks
        token_obs_masks: list[list[bool]] = []
        token_sem_types: list[list[str]] = []

        for b_idx, (char_mask, char_types, offsets) in enumerate(
            zip(char_obs_masks, char_sem_types, offset_maps)
        ):
            tok_mask: list[bool] = []
            tok_types: list[str] = []
            for start, end in offsets.tolist():
                if start == end == 0:
                    # Padding or special token
                    tok_mask.append(False)
                    tok_types.append("padding")
                else:
                    char_idx = start
                    is_obs = char_mask[char_idx] if char_idx < len(char_mask) else False
                    sem_type = char_types[char_idx] if char_idx < len(char_types) else "unknown"
                    tok_mask.append(is_obs)
                    tok_types.append(sem_type)
            token_obs_masks.append(tok_mask)
            token_sem_types.append(tok_types)

        yield EncodedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            observation_masks=token_obs_masks,
            semantic_types=token_sem_types,
            trajectory_ids=traj_ids,
            prompt_ids=prompt_ids,
        )
