"""LaTeX and text table generation for CCSM Phase 1 results."""

from __future__ import annotations

from typing import Optional

from ccsm_eval.analysis.absolute_surprise import AbsoluteSurpriseResult
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
        r"\begin{tabular}{llllllllrrrrr}",
        r"\toprule",
        r"Env & Model & Format & Prompt & Quality & Surprise & Semantic & $\rho$ & $p$ & CI$_{95}$ & Perm-$p$ & $n$ \\",
        r"\midrule",
    ]

    for r in results:
        ci = f"[{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]"
        fmt = getattr(r, "format_id", "")
        lines.append(
            f"{r.environment} & {r.model_id} & {fmt} & {r.prompt_id} & {r.quality_metric} & "
            f"{r.surprise_type} & {r.semantic_filter} & {_fmt(r.rho)} & {_fmt(r.p_value)} & "
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


def absolute_surprise_text(
    results: list[AbsoluteSurpriseResult],
    prompt_filter: Optional[str] = None,
) -> str:
    """Return a plain-text absolute surprise table.

    Args:
        results:       Output of compute_absolute_surprise().
        prompt_filter: If set, only show rows for this prompt_id. Useful for
                       limiting output in the main report (e.g. one canonical
                       prompt per environment).
    """
    if not results:
        return ""

    if prompt_filter:
        rows = [r for r in results if r.prompt_id == prompt_filter]
    else:
        rows = list(results)

    if not rows:
        return ""

    def _nan(v: float, fmt: str) -> str:
        return "--" if (v != v) else format(v, fmt)  # NaN check

    header = (
        f"  {'env':14s} | {'model':20s} | {'fmt':8s} | {'prompt':12s} | "
        f"{'semantic':20s} | {'mean_surp':>10s}  {'perplexity':>10s}  "
        f"{'range':>8s}  {'s/n':>6s}  {'n_tok':>8s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [header, sep]

    for r in rows:
        fmt_col = r.format_id if r.format_id else "-"
        env_col = f"{r.environment}({r.format_id})" if r.format_id else r.environment
        lines.append(
            f"  {env_col:14s} | {r.model_id:20s} | {fmt_col:8s} | {r.prompt_id:12s} | "
            f"{r.semantic_type:20s} | {r.mean_surprise:>10.2f}  "
            f"{r.perplexity:>10.0f}  "
            f"{_nan(r.surprise_range, '8.2f'):>8s}  "
            f"{_nan(r.signal_to_noise, '6.2f'):>6s}  "
            f"{r.n_tokens:>8d}"
        )

    return "\n".join(lines)


def absolute_surprise_latex(
    results: list[AbsoluteSurpriseResult],
    caption: str = "Absolute Surprise Levels by Domain and Semantic Type",
    label: str = "tab:absolute_surprise",
    prompt_filter: Optional[str] = None,
) -> str:
    """Return a LaTeX table of absolute surprise metrics."""
    if not results:
        return ""

    rows = [r for r in results if not prompt_filter or r.prompt_id == prompt_filter]
    if not rows:
        return ""

    def _nan(v: float, fmt: str) -> str:
        return "--" if (v != v) else format(v, fmt)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lllllrrrr}",
        r"\toprule",
        r"Env & Model & Fmt & Prompt & Semantic & Mean $s_t$ & PPL & Range & S/N \\",
        r"\midrule",
    ]

    for r in rows:
        fmt_col = r.format_id if r.format_id else "--"
        ppl_str = f"{r.perplexity:.0f}"
        lines.append(
            f"{r.environment} & {r.model_id} & {fmt_col} & {r.prompt_id} & "
            f"{r.semantic_type} & {r.mean_surprise:.2f} & {ppl_str} & "
            f"{_nan(r.surprise_range, '.2f')} & {_nan(r.signal_to_noise, '.2f')} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_text_report(all_results: dict, metadata: Optional[dict] = None) -> str:
    """Generate a human-readable plain-text summary of the evaluation."""
    lines = ["=" * 70, "CCSM Phase 1: Prior Quality Validation — Summary", "=" * 70, ""]

    # Embed run metadata at the top so the file is self-describing
    if metadata:
        lines.append("RUN INFO")
        lines.append("-" * 40)
        for key in ("run_id", "timestamp", "exp_name", "env_type", "fast_iteration", "seed"):
            if key in metadata:
                lines.append(f"  {key}: {metadata[key]}")
        if metadata.get("models_evaluated_fresh"):
            lines.append(f"  models (fresh):      {metadata['models_evaluated_fresh']}")
        if metadata.get("models_loaded_from_checkpoint"):
            lines.append(f"  models (checkpoint): {metadata['models_loaded_from_checkpoint']}")
        lines.append(f"  prompts: {metadata.get('prompts', [])}")
        lines.append("")

    abs_results = all_results.get("absolute_surprise_results", [])
    if abs_results:
        lines.append("ABSOLUTE SURPRISE LEVELS")
        lines.append("-" * 40)
        # Pick first prompt seen as canonical display prompt
        first_prompt = abs_results[0].prompt_id if abs_results else None
        lines.append(absolute_surprise_text(abs_results, prompt_filter=first_prompt))
        lines.append("")

    corr_results = all_results.get("correlation_results", [])
    if corr_results:
        lines.append("CORRELATION RESULTS")
        lines.append("-" * 40)
        lines.append(
            f"  {'':2s} {'env':12s} | {'model':20s} | {'format':10s} | {'prompt':12s} | "
            f"{'quality':20s} | {'surprise':12s} | {'semantic':18s} | "
            f"{'ρ':>7s}  {'p':>7s}  CI"
        )
        lines.append("  " + "-" * 130)
        for r in corr_results:
            sig = "**" if r.p_value < 0.05 else "  "
            fmt = getattr(r, "format_id", "")
            lines.append(
                f"  {sig} {r.environment:12s} | {r.model_id:20s} | {fmt:10s} | {r.prompt_id:12s} | "
                f"{r.quality_metric:20s} | {r.surprise_type:12s} | {r.semantic_filter:18s} | "
                f"ρ={r.rho:+.3f}  p={r.p_value:.3f}  "
                f"[{r.ci_low:+.3f},{r.ci_high:+.3f}]"
            )
        lines.append("")

    cf_results = all_results.get("counterfactual_results", [])
    if cf_results:
        lines.append("COUNTERFACTUAL RESULTS")
        lines.append("-" * 40)
        for r in cf_results:
            lines.append(
                f"  {r.environment:12s} | {r.model_id:20s} | {r.prompt_id:12s} | dir={r.direction} | "
                f"sign_consistency={r.sign_consistency:.1%} "
                f"effect_size={r.effect_size:.2f} "
                f"ρ(ΔQ,ΔS)={r.rho_quality_vs_delta_surprise:+.3f}"
            )
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
