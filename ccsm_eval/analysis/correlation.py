"""Correlation analysis: Spearman ρ between quality and surprise.

Implements:
    - Spearman ρ with bootstrap confidence intervals
    - Permutation null distribution
    - Per-semantic-type stratified correlation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from ccsm_eval.trajectories.base import SurpriseResult


@dataclass
class CorrelationResult:
    """Output of a single quality–surprise correlation test."""

    prompt_id: str
    model_id: str
    format_id: str                  # e.g. "fen", "natural", "templated"
    environment: str
    quality_metric: str            # e.g. "centipawn_eval", "q_pareto"
    surprise_type: str             # "cumulative", "normalised", "residualised"
    semantic_filter: str           # "all", "opponent_move", "board_state", ...
    n_trajectories: int
    rho: float                     # Spearman ρ
    p_value: float                 # two-tailed p-value
    ci_low: float                  # 95% bootstrap CI lower bound
    ci_high: float                 # 95% bootstrap CI upper bound
    permutation_p: float           # fraction of permutation ρ values ≥ |ρ_obs|
    permutation_mean: float
    permutation_std: float
    quality_values: list[float] = field(default_factory=list)
    surprise_values: list[float] = field(default_factory=list)


def compute_correlation(
    results: list[SurpriseResult],
    quality_metric: str,
    surprise_type: str = "normalised",
    semantic_filter: str = "all",
    bootstrap_n: int = 1000,
    permutation_n: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Optional[CorrelationResult]:
    """Compute Spearman ρ between quality and surprise with controls.

    Args:
        results:         List of SurpriseResult objects (same prompt + model).
        quality_metric:  Key into SurpriseResult.quality_scores.
        surprise_type:   One of "cumulative", "normalised", "residualised",
                         "stratified_<sem_type>".
        semantic_filter: Restrict to specific semantic token types; "all" uses all.
        bootstrap_n:     Number of bootstrap resamples for CIs.
        permutation_n:   Number of permutations for null distribution.
        confidence_level: CI level (default 0.95 → 95%).
        seed:            RNG seed for bootstrap and permutation.

    Returns:
        CorrelationResult or None if fewer than 3 trajectories have the metric.
    """
    # Extract quality and surprise values
    quality_vals: list[float] = []
    surprise_vals: list[float] = []

    for r in results:
        q = r.quality_scores.get(quality_metric)
        if q is None:
            continue
        s = _extract_surprise(r, surprise_type, semantic_filter)
        if s is None or not np.isfinite(s):
            continue
        quality_vals.append(float(q))
        surprise_vals.append(s)

    if len(quality_vals) < 3:
        return None

    q_arr = np.array(quality_vals)
    s_arr = np.array(surprise_vals)

    # Spearman ρ
    rho, p_value = stats.spearmanr(q_arr, s_arr)

    # Bootstrap confidence intervals
    rng = np.random.default_rng(seed)
    bootstrap_rhos = _bootstrap_rho(q_arr, s_arr, bootstrap_n, rng)
    alpha = 1 - confidence_level
    ci_low = float(np.percentile(bootstrap_rhos, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2)))

    # Permutation null
    perm_rhos = _permutation_null(q_arr, s_arr, permutation_n, rng)
    permutation_p = float(np.mean(np.abs(perm_rhos) >= abs(rho)))
    permutation_mean = float(np.mean(perm_rhos))
    permutation_std = float(np.std(perm_rhos))

    prompt_id = results[0].prompt_id if results else ""
    model_id = results[0].model_id if results else ""
    format_id = results[0].metadata.get("format", "") if results else ""
    environment = results[0].environment if results else ""

    return CorrelationResult(
        prompt_id=prompt_id,
        model_id=model_id,
        format_id=format_id,
        environment=environment,
        quality_metric=quality_metric,
        surprise_type=surprise_type,
        semantic_filter=semantic_filter,
        n_trajectories=len(quality_vals),
        rho=float(rho),
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
        permutation_p=permutation_p,
        permutation_mean=permutation_mean,
        permutation_std=permutation_std,
        quality_values=quality_vals,
        surprise_values=surprise_vals,
    )


def _extract_surprise(
    result: SurpriseResult, surprise_type: str, semantic_filter: str
) -> Optional[float]:
    """Extract the appropriate surprise value from a SurpriseResult."""
    if surprise_type == "cumulative":
        if semantic_filter == "all":
            return result.cumulative_surprise
        return _stratified_surprise(result, semantic_filter, normalise=False)

    elif surprise_type == "normalised":
        if semantic_filter == "all":
            return result.normalised_surprise
        return _stratified_surprise(result, semantic_filter, normalise=True)

    elif surprise_type == "residualised":
        return _residualised_surprise(result, semantic_filter)

    elif surprise_type.startswith("stratified_"):
        sem_type = surprise_type[len("stratified_"):]
        return _stratified_surprise(result, sem_type, normalise=True)

    return None


def _stratified_surprise(
    result: SurpriseResult, sem_type: str, normalise: bool
) -> Optional[float]:
    """Compute mean surprise over tokens of a specific semantic type."""
    values = [
        s
        for s, t in zip(result.per_token_surprise, result.per_token_semantic_type)
        if t == sem_type
    ]
    if not values:
        return None
    return float(np.mean(values)) if normalise else float(sum(values))


def _residualised_surprise(result: SurpriseResult, semantic_filter: str) -> Optional[float]:
    """Regress per-token surprise on unigram logprob; return mean residual."""
    from statsmodels.api import OLS, add_constant

    surprises = result.per_token_surprise
    unigrams = result.per_token_unigram_logprob
    sem_types = result.per_token_semantic_type

    if semantic_filter != "all":
        idxs = [i for i, t in enumerate(sem_types) if t == semantic_filter]
        surprises = [surprises[i] for i in idxs]
        unigrams = [unigrams[i] for i in idxs]

    if len(surprises) < 3:
        return None

    y = np.array(surprises)
    x = add_constant(np.array(unigrams))

    try:
        model = OLS(y, x).fit()
        residuals = model.resid
        return float(np.mean(residuals))
    except Exception:
        # Fallback: simple mean deviation
        expected = np.array(unigrams)
        residuals = y - (expected - expected.mean() + y.mean())
        return float(np.mean(residuals))


def _bootstrap_rho(
    q: np.ndarray, s: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    rhos = np.empty(n)
    size = len(q)
    for i in range(n):
        idx = rng.integers(0, size, size=size)
        rho_b, _ = stats.spearmanr(q[idx], s[idx])
        rhos[i] = rho_b
    return rhos


def _permutation_null(
    q: np.ndarray, s: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    rhos = np.empty(n)
    q_perm = q.copy()
    for i in range(n):
        rng.shuffle(q_perm)
        rho_p, _ = stats.spearmanr(q_perm, s)
        rhos[i] = rho_p
    return rhos
