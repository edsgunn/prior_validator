"""Failure mode analysis: trajectories where surprise and quality disagree.

Identifies cases where:
    - High quality but high surprise ("model doesn't recognise good play")
    - Low quality but low surprise ("model treats bad play as typical")

And classifies these by potential confound cause:
    - style_over_substance: low-quality with high-frequency tokens
    - notation_confusion:   high-quality with unusual formatting
    - history_dependence:   surprise diverges from quality late in trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ccsm_eval.trajectories.base import SurpriseResult


@dataclass
class FailureCase:
    """A trajectory where surprise and quality disagree."""

    trajectory_id: str
    quality_level: str
    environment: str
    prompt_id: str
    quality_value: float
    surprise_value: float
    disagreement_type: str         # "high_q_high_s" or "low_q_low_s"
    potential_cause: str           # "style_over_substance", "notation_confusion",
                                   #  "history_dependence", "unknown"
    evidence: dict = field(default_factory=dict)


def identify_failure_modes(
    results: list[SurpriseResult],
    quality_metric: str,
    top_n: int = 20,
) -> list[FailureCase]:
    """Extract and classify trajectories with high quality–surprise disagreement.

    Args:
        results:         SurpriseResult list (same prompt + model).
        quality_metric:  Which quality metric to use.
        top_n:           How many failure cases to return (by disagreement severity).

    Returns:
        List of FailureCase objects sorted by disagreement severity.
    """
    valid = [
        r for r in results
        if r.quality_scores.get(quality_metric) is not None
        and len(r.per_token_surprise) > 0
    ]

    if not valid:
        return []

    quality_vals = np.array([r.quality_scores[quality_metric] for r in valid])
    surprise_vals = np.array([r.normalised_surprise for r in valid])

    # Standardise both to [0, 1] range for comparison
    q_std = _safe_normalize(quality_vals)
    s_std = _safe_normalize(surprise_vals)

    cases: list[FailureCase] = []

    for i, r in enumerate(valid):
        q_norm = q_std[i]
        s_norm = s_std[i]

        # Disagreement score: high quality + high surprise, or low quality + low surprise
        # "High quality, high surprise" — surprise should be LOW
        if q_norm > 0.6 and s_norm > 0.6:
            disagreement_type = "high_q_high_s"
            severity = q_norm * s_norm
        # "Low quality, low surprise" — surprise should be HIGH
        elif q_norm < 0.4 and s_norm < 0.4:
            disagreement_type = "low_q_low_s"
            severity = (1 - q_norm) * (1 - s_norm)
        else:
            continue

        cause, evidence = _classify_cause(r)
        cases.append(
            FailureCase(
                trajectory_id=r.trajectory_id,
                quality_level=r.quality_level,
                environment=r.environment,
                prompt_id=r.prompt_id,
                quality_value=float(quality_vals[i]),
                surprise_value=float(surprise_vals[i]),
                disagreement_type=disagreement_type,
                potential_cause=cause,
                evidence=evidence,
            )
        )

    # Sort by severity (severity is already computed as product of norms)
    cases.sort(key=lambda c: abs(c.quality_value - c.surprise_value), reverse=True)
    return cases[:top_n]


def _classify_cause(result: SurpriseResult) -> tuple[str, dict]:
    """Classify the likely cause of a failure case using heuristics."""
    evidence: dict = {}

    # 1. Style-over-substance: mean unigram logprob is high → low surprise from common tokens
    if result.per_token_unigram_logprob:
        mean_unigram = float(np.mean(result.per_token_unigram_logprob))
        evidence["mean_unigram_logprob"] = mean_unigram
        # Unigram logprobs are negative; values closer to 0 = more common tokens
        if mean_unigram > -3.0:  # threshold: high token frequency
            return "style_over_substance", evidence

    # 2. Notation confusion: surprise is dominated by a few extreme tokens
    if result.per_token_surprise:
        surprises = np.array(result.per_token_surprise)
        cv = float(np.std(surprises) / (np.mean(surprises) + 1e-8))
        evidence["surprise_cv"] = cv
        if cv > 2.0:  # High coefficient of variation = spiky distribution
            return "notation_confusion", evidence

    # 3. History dependence: surprise increases late in trajectory
    if len(result.per_token_surprise) >= 10:
        surprises = np.array(result.per_token_surprise)
        first_half = float(np.mean(surprises[: len(surprises) // 2]))
        second_half = float(np.mean(surprises[len(surprises) // 2 :]))
        evidence["first_half_mean"] = first_half
        evidence["second_half_mean"] = second_half
        if second_half > first_half * 1.5:
            return "history_dependence", evidence

    return "unknown", evidence


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def summarise_failure_modes(cases: list[FailureCase]) -> dict[str, int]:
    """Count failure cases by type and cause."""
    summary: dict[str, int] = {}
    for c in cases:
        key = f"{c.disagreement_type}/{c.potential_cause}"
        summary[key] = summary.get(key, 0) + 1
    return summary
