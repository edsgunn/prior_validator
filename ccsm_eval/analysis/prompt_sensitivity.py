"""Prompt sensitivity analysis.

Measures how much the quality–surprise correlation varies across character prompts.
Low variance = robust prior; high variance = fragile, prompt-engineering is load-bearing.

Also implements the key GM vs. beginner comparison for chess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from ccsm_eval.analysis.correlation import CorrelationResult


@dataclass
class PromptSensitivityResult:
    """Cross-prompt sensitivity analysis for a single environment + model."""

    model_id: str
    environment: str
    quality_metric: str
    surprise_type: str

    prompt_rhos: dict[str, float]          # prompt_id -> Spearman ρ
    prompt_p_values: dict[str, float]      # prompt_id -> p-value
    mean_rho: float
    std_rho: float                         # Variance across "reasonable" prompts
    min_rho: float
    max_rho: float

    # Key comparison: aligned vs. inverted prompt
    aligned_prompt: Optional[str]
    mismatched_prompt: Optional[str]
    rho_delta: Optional[float]             # rho(aligned) - rho(mismatched)

    # Surprising inversion: does the "inverted" prompt (e.g. beginner) show
    # opposite sign of rho?
    inverted_prompt: Optional[str]
    rho_inversion: Optional[float]         # rho(aligned) - rho(inverted)
    inversion_significant: bool


def analyse_prompt_sensitivity(
    correlation_results: list[CorrelationResult],
    aligned_prompt: Optional[str] = None,
    mismatched_prompt: Optional[str] = None,
    inverted_prompt: Optional[str] = None,
) -> PromptSensitivityResult:
    """Compute cross-prompt sensitivity from a list of correlation results.

    Args:
        correlation_results: Results from the same environment + model, varying prompts.
        aligned_prompt:      Prompt expected to give the strongest negative ρ
                             (e.g. "GM" for chess).
        mismatched_prompt:   Irrelevant prompt expected to give weak ρ
                             (e.g. "mismatched").
        inverted_prompt:     Prompt expected to give inverted ρ
                             (e.g. "beginner" for chess).

    Returns:
        PromptSensitivityResult.
    """
    if not correlation_results:
        raise ValueError("No correlation results provided.")

    prompt_rhos = {r.prompt_id: r.rho for r in correlation_results}
    prompt_ps = {r.prompt_id: r.p_value for r in correlation_results}

    # Exclude mismatched from variance calculation
    reasonable_rhos = [
        r.rho for r in correlation_results if r.prompt_id != mismatched_prompt
    ]

    rho_delta = None
    if aligned_prompt and mismatched_prompt:
        a_rho = prompt_rhos.get(aligned_prompt, float("nan"))
        m_rho = prompt_rhos.get(mismatched_prompt, float("nan"))
        if np.isfinite(a_rho) and np.isfinite(m_rho):
            rho_delta = a_rho - m_rho

    rho_inversion = None
    inversion_significant = False
    if aligned_prompt and inverted_prompt:
        a_rho = prompt_rhos.get(aligned_prompt, float("nan"))
        i_rho = prompt_rhos.get(inverted_prompt, float("nan"))
        if np.isfinite(a_rho) and np.isfinite(i_rho):
            rho_inversion = a_rho - i_rho
            # Inversion is "significant" if they have opposite signs
            inversion_significant = (a_rho < 0) and (i_rho > 0)

    model_id = correlation_results[0].model_id
    environment = correlation_results[0].environment
    quality_metric = correlation_results[0].quality_metric
    surprise_type = correlation_results[0].surprise_type

    return PromptSensitivityResult(
        model_id=model_id,
        environment=environment,
        quality_metric=quality_metric,
        surprise_type=surprise_type,
        prompt_rhos=prompt_rhos,
        prompt_p_values=prompt_ps,
        mean_rho=float(np.mean(reasonable_rhos)) if reasonable_rhos else float("nan"),
        std_rho=float(np.std(reasonable_rhos)) if reasonable_rhos else float("nan"),
        min_rho=float(np.min(reasonable_rhos)) if reasonable_rhos else float("nan"),
        max_rho=float(np.max(reasonable_rhos)) if reasonable_rhos else float("nan"),
        aligned_prompt=aligned_prompt,
        mismatched_prompt=mismatched_prompt,
        rho_delta=rho_delta,
        inverted_prompt=inverted_prompt,
        rho_inversion=rho_inversion,
        inversion_significant=inversion_significant,
    )
