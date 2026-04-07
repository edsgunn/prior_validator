"""Cross-model-size scaling analysis.

Tests whether quality–surprise correlation improves with model scale,
and whether prior quality is a general model property (correlated across domains).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from ccsm_eval.analysis.correlation import CorrelationResult


@dataclass
class ScalingResult:
    """Analysis of ρ as a function of model size."""

    environment: str
    quality_metric: str
    prompt_id: str

    model_sizes: list[float]       # Parameter counts (in billions)
    model_ids: list[str]
    rhos: list[float]              # Corresponding Spearman ρ values

    # Spearman ρ between model_size and quality-surprise ρ
    scaling_rho: float
    scaling_p: float

    # Is the relationship monotonically increasing?
    is_monotone: bool


@dataclass
class DomainTransferResult:
    """Tests whether per-model ρ is correlated across environments."""

    model_ids: list[str]
    env_a: str
    env_b: str
    rhos_a: list[float]
    rhos_b: list[float]
    transfer_rho: float            # Spearman ρ across models
    transfer_p: float


def analyse_scaling(
    results_by_model: dict[str, list[CorrelationResult]],
    model_sizes_b: dict[str, float],     # model_id -> size in billions
    environment: str,
    quality_metric: str,
    prompt_id: str,
) -> Optional[ScalingResult]:
    """Compute the relationship between model size and ρ for one env + prompt.

    Args:
        results_by_model: {model_id: [CorrelationResult, ...]}
        model_sizes_b:    {model_id: size_in_billions}
        environment:      Filter results to this environment.
        quality_metric:   Filter results to this quality metric.
        prompt_id:        Filter results to this prompt.

    Returns:
        ScalingResult or None if fewer than 2 models have data.
    """
    model_ids = []
    sizes = []
    rhos = []

    for model_id, corr_list in results_by_model.items():
        matching = [
            r for r in corr_list
            if r.environment == environment
            and r.quality_metric == quality_metric
            and r.prompt_id == prompt_id
        ]
        if not matching:
            continue
        # Take the first matching result (or the best one)
        best = min(matching, key=lambda r: r.rho)  # most negative ρ
        model_ids.append(model_id)
        sizes.append(model_sizes_b.get(model_id, float("nan")))
        rhos.append(best.rho)

    if len(model_ids) < 2:
        return None

    size_arr = np.array(sizes)
    rho_arr = np.array(rhos)
    valid = np.isfinite(size_arr) & np.isfinite(rho_arr)
    size_arr = size_arr[valid]
    rho_arr = rho_arr[valid]
    model_ids = [m for m, v in zip(model_ids, valid) if v]

    if len(size_arr) < 2:
        return None

    # Sort by model size
    order = np.argsort(size_arr)
    size_arr = size_arr[order]
    rho_arr = rho_arr[order]
    model_ids = [model_ids[i] for i in order]

    scaling_rho, scaling_p = stats.spearmanr(size_arr, rho_arr)

    # Check monotonicity (more negative ρ = better)
    is_monotone = all(
        rho_arr[i] <= rho_arr[i + 1] for i in range(len(rho_arr) - 1)
    )

    return ScalingResult(
        environment=environment,
        quality_metric=quality_metric,
        prompt_id=prompt_id,
        model_sizes=size_arr.tolist(),
        model_ids=model_ids,
        rhos=rho_arr.tolist(),
        scaling_rho=float(scaling_rho),
        scaling_p=float(scaling_p),
        is_monotone=is_monotone,
    )


def analyse_domain_transfer(
    results_by_model: dict[str, list[CorrelationResult]],
    env_a: str,
    env_b: str,
    quality_metric_a: str,
    quality_metric_b: str,
    prompt_id: str,
) -> Optional[DomainTransferResult]:
    """Test whether ρ in environment A is correlated with ρ in environment B.

    A strong cross-domain correlation means "prior quality" is a general property.
    """
    model_ids = []
    rhos_a = []
    rhos_b = []

    for model_id, corr_list in results_by_model.items():
        a_match = [
            r for r in corr_list
            if r.environment == env_a
            and r.quality_metric == quality_metric_a
            and r.prompt_id == prompt_id
        ]
        b_match = [
            r for r in corr_list
            if r.environment == env_b
            and r.quality_metric == quality_metric_b
            and r.prompt_id == prompt_id
        ]
        if not a_match or not b_match:
            continue
        model_ids.append(model_id)
        rhos_a.append(min(r.rho for r in a_match))
        rhos_b.append(min(r.rho for r in b_match))

    if len(model_ids) < 3:
        return None

    transfer_rho, transfer_p = stats.spearmanr(rhos_a, rhos_b)

    return DomainTransferResult(
        model_ids=model_ids,
        env_a=env_a,
        env_b=env_b,
        rhos_a=rhos_a,
        rhos_b=rhos_b,
        transfer_rho=float(transfer_rho),
        transfer_p=float(transfer_p),
    )
