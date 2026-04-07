"""Figure generation for CCSM Phase 1 evaluation results.

Produces publication-quality plots using matplotlib and seaborn.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from ccsm_eval.analysis.correlation import CorrelationResult
from ccsm_eval.analysis.counterfactual import CounterfactualAnalysisResult
from ccsm_eval.analysis.scaling import ScalingResult
from ccsm_eval.trajectories.base import SurpriseResult

# Global style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
_PALETTE = sns.color_palette("muted")

QUALITY_ORDER = ["optimal", "strong", "moderate", "weak", "random",
                 "cooperative_suboptimal", "greedy_successful",
                 "aggressive_failed", "incoherent",
                 "near_optimal", "wandering", "lost", "adversarial"]


def plot_quality_vs_surprise(
    results: list[SurpriseResult],
    quality_metric: str,
    surprise_type: str = "normalised",
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of quality vs. surprise with quality-level colouring.

    Args:
        results:         SurpriseResult list (same prompt + model).
        quality_metric:  Key into quality_scores.
        surprise_type:   "normalised" or "cumulative".
        title:           Plot title.
        output_path:     If set, save figure to this path.
    """
    valid = [
        r for r in results
        if r.quality_scores.get(quality_metric) is not None
    ]
    if not valid:
        raise ValueError("No valid results for plotting.")

    quality_vals = [r.quality_scores[quality_metric] for r in valid]
    if surprise_type == "normalised":
        surprise_vals = [r.normalised_surprise for r in valid]
        ylabel = "Normalised surprise (nats / obs. token)"
    else:
        surprise_vals = [r.cumulative_surprise for r in valid]
        ylabel = "Cumulative surprise (nats)"

    levels = [r.quality_level for r in valid]
    unique_levels = sorted(set(levels), key=lambda x: QUALITY_ORDER.index(x) if x in QUALITY_ORDER else 99)
    palette = {lv: _PALETTE[i % len(_PALETTE)] for i, lv in enumerate(unique_levels)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for lv in unique_levels:
        idxs = [i for i, l in enumerate(levels) if l == lv]
        ax.scatter(
            [quality_vals[i] for i in idxs],
            [surprise_vals[i] for i in idxs],
            label=lv,
            color=palette[lv],
            alpha=0.6,
            s=30,
        )

    # Fit line
    q_arr = np.array(quality_vals)
    s_arr = np.array(surprise_vals)
    m, b = np.polyfit(q_arr, s_arr, 1)
    x_fit = np.linspace(q_arr.min(), q_arr.max(), 100)
    ax.plot(x_fit, m * x_fit + b, "k--", linewidth=1.2, label="Linear fit")

    ax.set_xlabel(quality_metric.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Quality vs. Surprise ({surprise_type})")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_surprise_by_quality_level(
    results: list[SurpriseResult],
    surprise_type: str = "normalised",
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Box/violin plot of surprise distributions per quality level."""
    if not results:
        raise ValueError("No results for plotting.")

    levels = []
    surprises = []
    for r in results:
        if surprise_type == "normalised":
            s = r.normalised_surprise
        else:
            s = r.cumulative_surprise
        levels.append(r.quality_level)
        surprises.append(s)

    unique_levels = sorted(set(levels), key=lambda x: QUALITY_ORDER.index(x) if x in QUALITY_ORDER else 99)

    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_level = {lv: [s for s, l in zip(surprises, levels) if l == lv] for lv in unique_levels}

    positions = list(range(len(unique_levels)))
    box_data = [data_by_level[lv] for lv in unique_levels]
    ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=_PALETTE[0], alpha=0.5))

    ax.set_xticks(positions)
    ax.set_xticklabels([lv.replace("_", "\n") for lv in unique_levels], fontsize=9)
    ax.set_ylabel("Normalised surprise" if surprise_type == "normalised" else "Cumulative surprise")
    ax.set_title(title or "Surprise by Quality Level")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rho_by_prompt(
    corr_results: list[CorrelationResult],
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of Spearman ρ for each character prompt, with 95% CIs."""
    if not corr_results:
        raise ValueError("No correlation results for plotting.")

    prompt_ids = [r.prompt_id for r in corr_results]
    rhos = [r.rho for r in corr_results]
    ci_low = [r.ci_low for r in corr_results]
    ci_high = [r.ci_high for r in corr_results]
    yerr_low = [r - lo for r, lo in zip(rhos, ci_low)]
    yerr_high = [hi - r for r, hi in zip(rhos, ci_high)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(prompt_ids))
    bars = ax.bar(x, rhos, color=[_PALETTE[i % len(_PALETTE)] for i in range(len(prompt_ids))],
                  alpha=0.7, width=0.5)
    ax.errorbar(x, rhos, yerr=[yerr_low, yerr_high], fmt="none", color="black",
                linewidth=1.5, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_ids, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Spearman ρ (quality vs. surprise)")
    ax.set_title(title or "ρ by Character Prompt")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scaling_curve(
    scaling_result: ScalingResult,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of ρ as a function of model size."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scaling_result.model_sizes, scaling_result.rhos, "o-",
            color=_PALETTE[0], linewidth=2, markersize=8)
    for size, rho, mid in zip(
        scaling_result.model_sizes, scaling_result.rhos, scaling_result.model_ids
    ):
        ax.annotate(mid, (size, rho), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Spearman ρ")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title or f"Scaling: ρ vs. Model Size ({scaling_result.environment})")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_counterfactual_deltas(
    cf_result: CounterfactualAnalysisResult,
    delta_surprises: list[float],
    quality_deltas: list[float],
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of Δ quality vs. Δ surprise for counterfactual edits."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(quality_deltas, delta_surprises, alpha=0.5, s=30, color=_PALETTE[1])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Δ quality (positive = improvement)")
    ax.set_ylabel("Δ surprise (replacement − original)")
    ax.set_title(title or f"Counterfactual: Δ quality vs. Δ surprise\n"
                           f"Sign consistency: {cf_result.sign_consistency:.1%}  "
                           f"ρ = {cf_result.rho_quality_vs_delta_surprise:.2f}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def save_all_figures(
    all_results: dict,
    output_dir: str,
) -> None:
    """Save a standard set of figures for a completed evaluation run.

    Args:
        all_results: Dict from run_eval.py containing 'surprise_results',
                     'correlation_results', 'scaling_results', etc.
        output_dir:  Directory to save figures into.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for env, env_data in all_results.get("by_environment", {}).items():
        for prompt_id, prompt_data in env_data.get("by_prompt", {}).items():
            sr = prompt_data.get("surprise_results", [])
            if sr:
                fig = plot_surprise_by_quality_level(sr, title=f"{env} / {prompt_id}")
                fig.savefig(
                    os.path.join(output_dir, f"{env}_{prompt_id}_surprise_by_level.png"),
                    dpi=150, bbox_inches="tight"
                )
                plt.close(fig)

    print(f"Figures saved to {output_dir}")
