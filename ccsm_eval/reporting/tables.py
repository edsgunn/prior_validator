"""LaTeX and text table generation for CCSM Phase 1 results."""

from __future__ import annotations

from typing import Optional

from ccsm_eval.analysis.correlation import CorrelationResult
from ccsm_eval.analysis.counterfactual import CounterfactualAnalysisResult
from ccsm_eval.analysis.prompt_sensitivity import PromptSensitivityResult
from ccsm_eval.analysis.scaling import ScalingResult


def _fmt(val: Optional[float], decimals: int = 3) -> str:
    if val is None or val != val:  # NaN check
        return "--"
    return f"{val:.{decimals}f}"


def correlation_table_latex(
    results: list[CorrelationResult],
    caption: str = "Quality–Surprise Correlation Results",
    label: str = "tab:correlation",
) -> str:
    """Generate a LaTeX table of correlation results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llllrrrrr}",
        r"\toprule",
        r"Env & Prompt & Quality & Surprise & $\rho$ & $p$ & CI$_{95}$ & Perm-$p$ & $n$ \\",
        r"\midrule",
    ]

    for r in results:
        ci = f"[{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]"
        lines.append(
            f"{r.environment} & {r.prompt_id} & {r.quality_metric} & "
            f"{r.surprise_type} & {_fmt(r.rho)} & {_fmt(r.p_value)} & "
            f"{ci} & {_fmt(r.permutation_p)} & {r.n_trajectories} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def prompt_sensitivity_table_latex(
    results: list[PromptSensitivityResult],
    caption: str = "Prompt Sensitivity",
    label: str = "tab:prompt_sensitivity",
) -> str:
    """Generate a LaTeX table summarising cross-prompt ρ variance."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Env & Model & Quality & $\bar\rho$ & $\sigma_\rho$ & $\Delta\rho_{\text{align}}$ & Inversion \\",
        r"\midrule",
    ]

    for r in results:
        inversion = "Yes" if r.inversion_significant else "No"
        lines.append(
            f"{r.environment} & {r.model_id} & {r.quality_metric} & "
            f"{_fmt(r.mean_rho)} & {_fmt(r.std_rho)} & "
            f"{_fmt(r.rho_delta)} & {inversion} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def counterfactual_table_latex(
    results: list[CounterfactualAnalysisResult],
    caption: str = "Counterfactual Edit Analysis",
    label: str = "tab:counterfactual",
) -> str:
    """Generate a LaTeX table of counterfactual analysis results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Env & Model & Prompt & Dir & Sign cons. & $\bar{\Delta S}$ & $\rho$ & Effect size \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"{r.environment} & {r.model_id} & {r.prompt_id} & {r.direction} & "
            f"{_fmt(r.sign_consistency, 2)} & {_fmt(r.mean_delta_surprise)} & "
            f"{_fmt(r.rho_quality_vs_delta_surprise)} & {_fmt(r.effect_size)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def scaling_table_latex(
    results: list[ScalingResult],
    caption: str = "Scaling Analysis",
    label: str = "tab:scaling",
) -> str:
    """Generate a LaTeX table of scaling results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lllrrl}",
        r"\toprule",
        r"Env & Prompt & Quality & Scaling $\rho$ & $p$ & Monotone \\",
        r"\midrule",
    ]

    for r in results:
        monotone = "Yes" if r.is_monotone else "No"
        lines.append(
            f"{r.environment} & {r.prompt_id} & {r.quality_metric} & "
            f"{_fmt(r.scaling_rho)} & {_fmt(r.scaling_p)} & {monotone} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summary_text_report(all_results: dict) -> str:
    """Generate a human-readable plain-text summary of the evaluation."""
    lines = ["=" * 70, "CCSM Phase 1: Prior Quality Validation — Summary", "=" * 70, ""]

    corr_results = all_results.get("correlation_results", [])
    if corr_results:
        lines.append("CORRELATION RESULTS")
        lines.append("-" * 40)
        for r in corr_results:
            sig = "**" if r.p_value < 0.05 else "  "
            lines.append(
                f"{sig} {r.environment:12s} | {r.prompt_id:15s} | "
                f"{r.quality_metric:20s} | "
                f"ρ={r.rho:+.3f} p={r.p_value:.3f} "
                f"CI=[{r.ci_low:+.3f},{r.ci_high:+.3f}]"
            )
        lines.append("")

    cf_results = all_results.get("counterfactual_results", [])
    if cf_results:
        lines.append("COUNTERFACTUAL RESULTS")
        lines.append("-" * 40)
        for r in cf_results:
            lines.append(
                f"  {r.environment:12s} | {r.prompt_id:15s} | "
                f"sign_consistency={r.sign_consistency:.1%} "
                f"effect_size={r.effect_size:.2f} "
                f"ρ(ΔQ,ΔS)={r.rho_quality_vs_delta_surprise:+.3f}"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
