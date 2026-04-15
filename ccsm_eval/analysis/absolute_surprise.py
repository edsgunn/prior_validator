"""Absolute surprise level (perplexity) analysis.

Computes baseline domain competence metrics from already-collected per-token
surprise data — no new model inference required.

Key outputs:
    mean_surprise  — mean per-token negative log-prob across all obs tokens
    perplexity     — exp(mean_surprise), standard definition
    surprise_range — max(mean_surprise per quality level) - min(...)
    signal_to_noise — surprise_range / std_surprise

A high perplexity means the model is close to guessing in this domain.
A low S/N means quality-correlated surprise variation is buried in noise.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ccsm_eval.trajectories.base import SurpriseResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AbsoluteSurpriseResult:
    environment: str
    model_id: str
    format_id: str          # "fen", "natural", "human", "" for single-format envs
    prompt_id: str
    semantic_type: str      # "all" or a specific type
    mean_surprise: float
    std_surprise: float
    perplexity: float
    surprise_range: float   # max - min mean_surprise across quality levels
    signal_to_noise: float  # surprise_range / std_surprise (nan if std == 0)
    n_tokens: int
    n_trajectories: int
    per_quality_mean: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_absolute_surprise(
    results: list[SurpriseResult],
    semantic_types: Optional[list[str]] = None,
) -> list[AbsoluteSurpriseResult]:
    """Compute absolute surprise metrics from existing SurpriseResult objects.

    Groups by (environment, model_id, format_id, prompt_id) and computes
    aggregate statistics over all observation tokens, plus per-semantic-type
    breakdowns.

    Args:
        results:        SurpriseResult objects from the evaluation pipeline.
        semantic_types: Semantic types to compute per-type stats for, in
                        addition to the "all" aggregate. If None, auto-detects
                        all types present in the data.

    Returns:
        List of AbsoluteSurpriseResult, one per (group × semantic_type).
    """
    if not results:
        return []

    # Auto-detect semantic types if not specified
    if semantic_types is None:
        seen: set[str] = set()
        for r in results:
            seen.update(r.per_token_semantic_type)
        semantic_types = sorted(seen)

    # Group results by (env, model, format, prompt)
    groups: dict[tuple, list[SurpriseResult]] = defaultdict(list)
    for r in results:
        fmt = r.metadata.get("format", "")
        key = (r.environment, r.model_id, fmt, r.prompt_id)
        groups[key].append(r)

    output: list[AbsoluteSurpriseResult] = []

    for (env, model, fmt, prompt), group in sorted(groups.items()):
        # Compute for "all" types and each individual type
        for sem_type in ["all"] + list(semantic_types):
            asr = _compute_group(env, model, fmt, prompt, sem_type, group)
            if asr is not None:
                output.append(asr)

    return output


def _compute_group(
    env: str,
    model: str,
    fmt: str,
    prompt: str,
    sem_type: str,
    group: list[SurpriseResult],
) -> Optional[AbsoluteSurpriseResult]:
    """Compute AbsoluteSurpriseResult for one group + semantic_type."""
    all_tokens: list[float] = []
    by_quality: dict[str, list[float]] = defaultdict(list)
    n_trajectories = len(group)

    for r in group:
        surprises = r.per_token_surprise
        types = r.per_token_semantic_type
        ql = r.quality_level

        if sem_type == "all":
            tokens = surprises
        else:
            tokens = [s for s, t in zip(surprises, types) if t == sem_type]

        if not tokens:
            continue

        all_tokens.extend(tokens)
        by_quality[ql].extend(tokens)

    if len(all_tokens) < 2:
        return None

    arr = np.array(all_tokens, dtype=np.float64)
    mean_s = float(np.mean(arr))
    std_s = float(np.std(arr))
    perp = math.exp(min(mean_s, 700))  # cap to avoid overflow

    # Per-quality-level means
    quality_means: dict[str, float] = {
        ql: float(np.mean(toks)) for ql, toks in by_quality.items() if toks
    }

    if len(quality_means) >= 2:
        vals = list(quality_means.values())
        surprise_range = float(max(vals) - min(vals))
    else:
        surprise_range = 0.0

    sn = surprise_range / std_s if std_s > 0 else float("nan")

    return AbsoluteSurpriseResult(
        environment=env,
        model_id=model,
        format_id=fmt,
        prompt_id=prompt,
        semantic_type=sem_type,
        mean_surprise=mean_s,
        std_surprise=std_s,
        perplexity=perp,
        surprise_range=surprise_range,
        signal_to_noise=sn,
        n_tokens=len(all_tokens),
        n_trajectories=n_trajectories,
        per_quality_mean=quality_means,
    )
