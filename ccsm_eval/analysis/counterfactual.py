"""Counterfactual surprise analysis.

Aggregates CounterfactualSurpriseResult objects and computes:
    - Sign consistency (fraction with predicted sign)
    - Effect size relative to background surprise variance
    - Δ surprise as a function of Δ quality
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ccsm_eval.trajectories.base import CounterfactualSurpriseResult


@dataclass
class CounterfactualAnalysisResult:
    """Aggregate statistics for a set of counterfactual edits."""

    prompt_id: str
    model_id: str
    environment: str
    direction: str                 # "up", "down", or "both"
    n_edits: int

    # Sign consistency: fraction where Δ surprise has the expected sign
    # Expected: "up" edit → Δ surprise < 0 (replacement is less surprising)
    #           "down" edit → Δ surprise > 0 (replacement is more surprising)
    sign_consistency: float        # [0, 1]

    mean_delta_surprise: float
    std_delta_surprise: float
    mean_quality_delta: float

    # Spearman ρ between quality_delta and delta_surprise
    rho_quality_vs_delta_surprise: float
    rho_p_value: float

    # Effect size: mean |Δ surprise| / std of per-token surprises
    effect_size: float


def analyse_counterfactuals(
    cf_results: list[CounterfactualSurpriseResult],
    background_surprise_std: float = 1.0,
) -> CounterfactualAnalysisResult:
    """Compute aggregate statistics for counterfactual edit results.

    Args:
        cf_results:              List of counterfactual surprise measurements.
        background_surprise_std: Std of per-token surprise values (for effect size).
                                 Should be precomputed from the full trajectory set.
    """
    if not cf_results:
        raise ValueError("No counterfactual results to analyse.")

    delta_surprises = np.array([r.delta_surprise for r in cf_results])
    quality_deltas = np.array([r.quality_delta for r in cf_results])
    directions = [r.direction for r in cf_results]

    # Sign consistency
    # "up" edit: quality improves (quality_delta > 0) → expect delta_surprise < 0
    # "down" edit: quality worsens (quality_delta < 0) → expect delta_surprise > 0
    correct_signs = 0
    for ds, qd, d in zip(delta_surprises, quality_deltas, directions):
        if d == "up" and ds < 0:
            correct_signs += 1
        elif d == "down" and ds > 0:
            correct_signs += 1
        elif d not in ("up", "down"):
            # Use quality_delta sign
            if (qd > 0 and ds < 0) or (qd < 0 and ds > 0):
                correct_signs += 1

    sign_consistency = correct_signs / len(cf_results)

    # Correlation between quality delta and surprise delta
    if len(cf_results) >= 3:
        rho, p_val = stats.spearmanr(quality_deltas, delta_surprises)
    else:
        rho, p_val = float("nan"), float("nan")

    # Effect size
    mean_abs_delta = float(np.mean(np.abs(delta_surprises)))
    effect_size = mean_abs_delta / background_surprise_std if background_surprise_std > 0 else 0.0

    # Infer direction label
    unique_dirs = set(directions)
    direction_label = unique_dirs.pop() if len(unique_dirs) == 1 else "both"

    return CounterfactualAnalysisResult(
        prompt_id=cf_results[0].prompt_id,
        model_id=cf_results[0].model_id,
        environment="",  # should be passed in if needed
        direction=direction_label,
        n_edits=len(cf_results),
        sign_consistency=sign_consistency,
        mean_delta_surprise=float(np.mean(delta_surprises)),
        std_delta_surprise=float(np.std(delta_surprises)),
        mean_quality_delta=float(np.mean(quality_deltas)),
        rho_quality_vs_delta_surprise=float(rho),
        rho_p_value=float(p_val),
        effect_size=effect_size,
    )
