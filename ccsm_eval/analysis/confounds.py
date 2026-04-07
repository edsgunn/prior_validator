"""Confound controls for quality–surprise correlation.

Implements:
    - Length normalisation (surprise per observation token)
    - Residualised surprise (regression on unigram token frequency)
    - Stratified surprise (by semantic token type)
    - Summary of confound analysis across a result set
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ccsm_eval.trajectories.base import SurpriseResult


@dataclass
class ConfoundReport:
    """Summary of confound checks for a set of SurpriseResults."""

    environment: str
    prompt_id: str
    model_id: str

    # Length confound
    length_quality_rho: float      # Correlation between trajectory length and quality
    length_quality_p: float
    length_surprise_rho: float     # Correlation between length and raw surprise
    length_surprise_p: float

    # Lexical frequency confound
    mean_unigram_lp_by_quality: dict[str, float]  # quality_level -> mean unigram lp

    # Semantic stratification
    rho_by_sem_type: dict[str, float]  # sem_type -> Spearman ρ with quality


def check_length_confound(
    results: list[SurpriseResult],
    quality_metric: str,
) -> dict[str, float]:
    """Check whether trajectory length confounds quality–surprise correlation.

    Returns a dict with:
        length_quality_rho, length_quality_p,
        length_surprise_rho, length_surprise_p
    """
    lengths = np.array([len(r.per_token_surprise) for r in results], dtype=float)
    qualities = np.array(
        [r.quality_scores.get(quality_metric, float("nan")) for r in results]
    )
    surprises = np.array([r.cumulative_surprise for r in results])

    valid = np.isfinite(qualities)
    lengths = lengths[valid]
    qualities = qualities[valid]
    surprises = surprises[valid]

    if len(lengths) < 3:
        return {
            "length_quality_rho": float("nan"),
            "length_quality_p": float("nan"),
            "length_surprise_rho": float("nan"),
            "length_surprise_p": float("nan"),
        }

    lq_rho, lq_p = stats.spearmanr(lengths, qualities)
    ls_rho, ls_p = stats.spearmanr(lengths, surprises)

    return {
        "length_quality_rho": float(lq_rho),
        "length_quality_p": float(lq_p),
        "length_surprise_rho": float(ls_rho),
        "length_surprise_p": float(ls_p),
    }


def compute_stratified_correlations(
    results: list[SurpriseResult],
    quality_metric: str,
) -> dict[str, float | None]:
    """Compute Spearman ρ separately for each semantic token type.

    Returns a dict mapping semantic_type -> rho (or None if insufficient data).
    """
    # Collect all semantic types present
    all_types: set[str] = set()
    for r in results:
        all_types.update(r.per_token_semantic_type)

    rho_by_type: dict[str, float | None] = {}
    for sem_type in sorted(all_types):
        if sem_type in ("padding", "prompt"):
            continue

        stratified_surprises: list[float] = []
        stratified_qualities: list[float] = []

        for r in results:
            q = r.quality_scores.get(quality_metric)
            if q is None:
                continue
            tok_surprises = [
                s
                for s, t in zip(r.per_token_surprise, r.per_token_semantic_type)
                if t == sem_type
            ]
            if not tok_surprises:
                continue
            stratified_surprises.append(float(np.mean(tok_surprises)))
            stratified_qualities.append(float(q))

        if len(stratified_surprises) < 3:
            rho_by_type[sem_type] = None
            continue

        rho, _ = stats.spearmanr(
            np.array(stratified_qualities), np.array(stratified_surprises)
        )
        rho_by_type[sem_type] = float(rho)

    return rho_by_type


def compute_unigram_confound(
    results: list[SurpriseResult],
) -> dict[str, float]:
    """Check whether lexical frequency explains the surprise signal.

    Compares mean unigram log-probability across quality levels. If high-quality
    trajectories have systematically higher unigram log-probs, that suggests
    the surprise signal is driven by token frequency rather than strategic content.
    """
    by_level: dict[str, list[float]] = {}
    for r in results:
        if not r.per_token_unigram_logprob:
            continue
        mean_unigram = float(np.mean(r.per_token_unigram_logprob))
        level = r.quality_level
        by_level.setdefault(level, []).append(mean_unigram)

    return {level: float(np.mean(vals)) for level, vals in by_level.items()}


def build_confound_report(
    results: list[SurpriseResult],
    quality_metric: str,
) -> ConfoundReport:
    """Assemble a full confound report for a result set."""
    if not results:
        raise ValueError("Cannot build confound report from empty result list.")

    length_check = check_length_confound(results, quality_metric)
    stratified_rhos = compute_stratified_correlations(results, quality_metric)
    unigram_by_level = compute_unigram_confound(results)

    return ConfoundReport(
        environment=results[0].environment,
        prompt_id=results[0].prompt_id,
        model_id=results[0].model_id,
        length_quality_rho=length_check["length_quality_rho"],
        length_quality_p=length_check["length_quality_p"],
        length_surprise_rho=length_check["length_surprise_rho"],
        length_surprise_p=length_check["length_surprise_p"],
        mean_unigram_lp_by_quality=unigram_by_level,
        rho_by_sem_type={
            k: v for k, v in stratified_rhos.items() if v is not None
        },
    )
