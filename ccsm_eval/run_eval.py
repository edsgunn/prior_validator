"""Main entry point for CCSM Phase 1: Prior Quality Validation.

Usage:
    python -m ccsm_eval.run_eval --config configs/chess_eval.yaml \\
                                 --models configs/models.yaml \\
                                 --output results/

    # Fast iteration (single model, two environments, two prompts each):
    python -m ccsm_eval.run_eval --config configs/chess_eval.yaml \\
                                 --models configs/models.yaml \\
                                 --fast-iteration

Pipeline:
    1. Load config and models.
    2. Generate trajectories (or load from cache).
    3. Score quality metrics.
    4. Format trajectories for each prompt.
    5. Run model forward passes → SurpriseResults.
    6. Run counterfactual edits → CounterfactualSurpriseResults.
    7. Compute correlations and confound checks.
    8. Compute prompt sensitivity and scaling analyses.
    9. Identify failure modes.
    10. Generate figures and tables.
    11. Write JSON results and text report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ccsm_eval")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Environment dispatch
# ---------------------------------------------------------------------------

def make_generator(env_config: dict):
    env_type = env_config["type"]
    if env_type == "chess":
        from ccsm_eval.trajectories.chess.generator import ChessTrajectoryGenerator
        return ChessTrajectoryGenerator(
            stockfish_path=env_config.get("stockfish_path", "stockfish"),
            white_time=env_config.get("white_time", 0.1),
            opponent_time=env_config.get("opponent_time", 0.05),
            max_moves=env_config.get("max_moves", 120),
        )
    elif env_type == "negotiation":
        from ccsm_eval.trajectories.negotiation.generator import NegotiationTrajectoryGenerator
        return NegotiationTrajectoryGenerator(
            max_rounds=env_config.get("max_rounds", 10),
        )
    elif env_type == "gridworld":
        from ccsm_eval.trajectories.gridworld.generator import GridworldTrajectoryGenerator
        return GridworldTrajectoryGenerator(
            min_size=env_config.get("min_size", 5),
            max_size=env_config.get("max_size", 8),
            wall_density=env_config.get("wall_density", 0.2),
            max_steps=env_config.get("max_steps", 100),
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}")


def make_quality_scorer(env_type: str, env_config: dict):
    if env_type == "chess":
        from ccsm_eval.trajectories.chess.quality import ChessQualityScorer
        return ChessQualityScorer(
            stockfish_path=env_config.get("stockfish_path", "stockfish"),
            eval_time=env_config.get("eval_time", 0.05),
        )
    elif env_type == "negotiation":
        from ccsm_eval.trajectories.negotiation.quality import NegotiationQualityScorer
        return NegotiationQualityScorer()
    elif env_type == "gridworld":
        from ccsm_eval.trajectories.gridworld.quality import GridworldQualityScorer
        return GridworldQualityScorer()
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}")


def make_formatter(env_type: str, fmt: str):
    if env_type == "chess":
        from ccsm_eval.trajectories.chess.formatter import make_chess_formatter
        return make_chess_formatter(fmt)
    elif env_type == "negotiation":
        from ccsm_eval.trajectories.negotiation.formatter import make_negotiation_formatter
        return make_negotiation_formatter(fmt)
    elif env_type == "gridworld":
        from ccsm_eval.trajectories.gridworld.formatter import GridworldFormatter
        return GridworldFormatter()
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}")


def make_counterfactual_editor(env_type: str, env_config: dict):
    cf_cfg = env_config.get("counterfactual", {})
    if env_type == "chess":
        from ccsm_eval.trajectories.chess.counterfactual import ChessCounterfactualEditor
        return ChessCounterfactualEditor(
            stockfish_path=env_config.get("stockfish_path", "stockfish"),
            eval_time=cf_cfg.get("eval_time", 0.05),
            opponent_time=env_config.get("opponent_time", 0.05),
            min_cp_delta=cf_cfg.get("min_quality_delta_cp", 100),
        )
    elif env_type == "negotiation":
        from ccsm_eval.trajectories.negotiation.counterfactual import NegotiationCounterfactualEditor
        return NegotiationCounterfactualEditor()
    elif env_type == "gridworld":
        from ccsm_eval.trajectories.gridworld.counterfactual import GridworldCounterfactualEditor
        return GridworldCounterfactualEditor()
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}")


# ---------------------------------------------------------------------------
# Main pipeline steps
# ---------------------------------------------------------------------------

def generate_and_score_trajectories(
    generator,
    scorer,
    quality_levels: list[str],
    counts: dict[str, int],
    seed: int,
    cache_path: Optional[str] = None,
) -> dict[str, list]:
    """Generate trajectories and compute quality scores. Cache to disk."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached trajectories from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    trajectories_by_level: dict[str, list] = {}
    for level in quality_levels:
        n = counts.get(level, 0)
        if n == 0:
            continue
        logger.info(f"  Generating {n} {level!r} trajectories ...")
        t0 = time.time()
        trajs = generator.generate(level, n, seed=seed + hash(level))

        # Score quality
        for traj in trajs:
            traj.quality_scores = scorer.score(traj)

        trajectories_by_level[level] = trajs
        logger.info(f"  Done ({time.time() - t0:.1f}s)")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(trajectories_by_level, f)
        logger.info(f"Cached trajectories to {cache_path}")

    return trajectories_by_level


def run_surprise_evaluation(
    evaluator,
    trajectories_by_level: dict[str, list],
    prompts: dict[str, str],
    formatters: dict[str, Any],
    env_type: str,
) -> list:
    """Format trajectories for each prompt and compute surprise."""
    from ccsm_eval.trajectories.base import SurpriseResult

    all_results: list[SurpriseResult] = []
    all_trajs = [t for trajs in trajectories_by_level.values() for t in trajs]

    for fmt_name, formatter in formatters.items():
        logger.info(f"  Formatting with {fmt_name!r} ...")
        items = []
        for traj in all_trajs:
            for prompt_id, prompt_text in prompts.items():
                text, obs_mask, sem_types = formatter.format(traj, prompt_text)
                items.append((traj, text, obs_mask, sem_types, prompt_id))

        logger.info(f"  Running {len(items)} forward passes ...")
        t0 = time.time()
        results = evaluator.evaluate_batch(items)
        logger.info(f"  Done ({time.time() - t0:.1f}s)")

        # Tag format in metadata
        for r in results:
            r.metadata["format"] = fmt_name

        all_results.extend(results)

    return all_results


def run_counterfactual_edits(
    editor,
    evaluator,
    trajectories_by_level: dict[str, list],
    prompts: dict[str, str],
    formatters: dict[str, Any],
    n_edits_per_traj: int,
    n_sample_trajectories: int,
    seed: int,
) -> list:
    """Run counterfactual edits on sampled trajectories."""
    import random
    from ccsm_eval.trajectories.base import CounterfactualSurpriseResult

    rng = random.Random(seed + 9999)
    all_cf_results: list[CounterfactualSurpriseResult] = []

    all_trajs = [t for trajs in trajectories_by_level.values() for t in trajs]
    sample_trajs = rng.sample(all_trajs, min(n_sample_trajectories, len(all_trajs)))

    formatter_name, formatter = next(iter(formatters.items()))

    for traj in sample_trajs:
        try:
            positions = editor.sample_edit_positions(traj, n_edits_per_traj, seed=rng.randint(0, 2**31))
        except Exception as e:
            logger.debug(f"Could not sample edit positions for {traj.trajectory_id}: {e}")
            continue

        for pos in positions:
            for direction in ("up", "down"):
                try:
                    edit = editor.edit(traj, pos, direction)
                except Exception as e:
                    logger.debug(f"Edit failed at pos {pos} dir {direction}: {e}")
                    continue

                for prompt_id, prompt_text in prompts.items():
                    try:
                        cf_result = evaluator.evaluate_counterfactual(
                            edit, traj, prompt_text, prompt_id, formatter
                        )
                        all_cf_results.append(cf_result)
                    except Exception as e:
                        logger.debug(f"CF evaluation failed: {e}")

    return all_cf_results


def run_analysis(
    surprise_results: list,
    cf_results: list,
    quality_metrics: list[str],
    surprise_types: list[str],
    semantic_filters: list[str],
    bootstrap_n: int,
    permutation_n: int,
    aligned_prompt: Optional[str],
    mismatched_prompt: Optional[str],
    inverted_prompt: Optional[str],
) -> dict:
    """Run all analysis stages and return results dict."""
    from ccsm_eval.analysis.correlation import compute_correlation
    from ccsm_eval.analysis.confounds import build_confound_report
    from ccsm_eval.analysis.counterfactual import analyse_counterfactuals
    from ccsm_eval.analysis.failure_modes import identify_failure_modes
    from ccsm_eval.analysis.prompt_sensitivity import analyse_prompt_sensitivity

    import numpy as np

    correlation_results = []
    confound_reports = []
    cf_analysis_results = []
    failure_cases = []
    prompt_sensitivity_results = []

    # Group by prompt + model
    by_prompt: dict[str, list] = {}
    for r in surprise_results:
        key = f"{r.prompt_id}::{r.model_id}"
        by_prompt.setdefault(key, []).append(r)

    for key, results_group in by_prompt.items():
        prompt_id, model_id = key.split("::", 1)

        for qm in quality_metrics:
            for st in surprise_types:
                for sf in semantic_filters:
                    corr = compute_correlation(
                        results_group,
                        quality_metric=qm,
                        surprise_type=st,
                        semantic_filter=sf,
                        bootstrap_n=bootstrap_n,
                        permutation_n=permutation_n,
                    )
                    if corr is not None:
                        correlation_results.append(corr)

            # Confound report (once per prompt/metric)
            try:
                report = build_confound_report(results_group, qm)
                confound_reports.append(report)
            except Exception as e:
                logger.debug(f"Confound report failed for {key}/{qm}: {e}")

            # Failure modes
            cases = identify_failure_modes(results_group, qm)
            failure_cases.extend(cases)

    # Counterfactual analysis
    if cf_results:
        by_prompt_cf: dict[str, list] = {}
        for r in cf_results:
            key = f"{r.prompt_id}::{r.model_id}"
            by_prompt_cf.setdefault(key, []).append(r)

        surprise_std = float(np.std([r.normalised_surprise for r in surprise_results])) or 1.0

        for key, cf_group in by_prompt_cf.items():
            try:
                cf_analysis = analyse_counterfactuals(cf_group, background_surprise_std=surprise_std)
                cf_analysis_results.append(cf_analysis)
            except Exception as e:
                logger.debug(f"CF analysis failed for {key}: {e}")

    # Prompt sensitivity
    by_model: dict[str, list] = {}
    for r in correlation_results:
        by_model.setdefault(r.model_id, []).append(r)

    for model_id, model_corrs in by_model.items():
        try:
            ps = analyse_prompt_sensitivity(
                model_corrs,
                aligned_prompt=aligned_prompt,
                mismatched_prompt=mismatched_prompt,
                inverted_prompt=inverted_prompt,
            )
            prompt_sensitivity_results.append(ps)
        except Exception as e:
            logger.debug(f"Prompt sensitivity failed for {model_id}: {e}")

    return {
        "correlation_results": correlation_results,
        "confound_reports": confound_reports,
        "counterfactual_results": cf_analysis_results,
        "failure_cases": failure_cases,
        "prompt_sensitivity_results": prompt_sensitivity_results,
    }


def save_results(results: dict, output_dir: str, name: str) -> None:
    """Save results to JSON and generate text report."""
    from ccsm_eval.reporting.tables import (
        correlation_table_latex,
        counterfactual_table_latex,
        summary_text_report,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Text summary
    report = summary_text_report(results)
    report_path = os.path.join(output_dir, f"{name}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(report)

    # LaTeX tables
    if results.get("correlation_results"):
        latex = correlation_table_latex(results["correlation_results"])
        with open(os.path.join(output_dir, f"{name}_correlations.tex"), "w") as f:
            f.write(latex)

    if results.get("counterfactual_results"):
        latex = counterfactual_table_latex(results["counterfactual_results"])
        with open(os.path.join(output_dir, f"{name}_counterfactuals.tex"), "w") as f:
            f.write(latex)

    # JSON results (serialise dataclasses to dicts)
    def _serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialise(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [_serialise(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    json_path = os.path.join(output_dir, f"{name}_results.json")
    with open(json_path, "w") as f:
        json.dump(_serialise(results), f, indent=2)
    logger.info(f"Results saved to {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CCSM Phase 1: Prior Quality Validation"
    )
    parser.add_argument("--config", required=True, help="Environment config YAML")
    parser.add_argument("--models", required=True, help="Models config YAML")
    parser.add_argument(
        "--output", default=None, help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--fast-iteration", action="store_true",
        help="Run the fast iteration subset (fewer trajectories, fewer prompts)"
    )
    parser.add_argument(
        "--skip-trajectories", action="store_true",
        help="Skip generation, load from cache"
    )
    parser.add_argument(
        "--skip-counterfactuals", action="store_true",
        help="Skip counterfactual edits"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed"
    )
    args = parser.parse_args()

    # Load configs
    env_cfg = load_config(args.config)
    models_cfg = load_config(args.models)

    seed = args.seed or env_cfg.get("experiment", {}).get("seed", 42)
    exp_name = env_cfg.get("experiment", {}).get("name", "eval")
    output_dir = args.output or env_cfg.get("experiment", {}).get("output_dir", "results")
    env_type = env_cfg["environment"]["type"]

    # Apply fast iteration overrides
    if args.fast_iteration:
        fi = models_cfg.get("fast_iteration", {})
        model_names = fi.get("models", [models_cfg["models"][0]["name"]])
        prompt_override = fi.get("prompts_per_env", {}).get(env_type)
        traj_count_override = fi.get("trajectory_counts", 20)
        logger.info(
            f"Fast iteration mode: models={model_names}, "
            f"traj_count={traj_count_override}"
        )
    else:
        model_names = [m["name"] for m in models_cfg["models"]]
        prompt_override = None
        traj_count_override = None

    # Environment components
    generator = make_generator(env_cfg["environment"])
    scorer = make_quality_scorer(env_type, env_cfg["environment"])
    formats = env_cfg["environment"].get("formats", ["fen"]) if env_type == "chess" else \
              env_cfg["environment"].get("formats", ["templated"]) if env_type == "negotiation" else \
              ["text"]
    formatters = {fmt: make_formatter(env_type, fmt) for fmt in formats}
    cf_editor = make_counterfactual_editor(env_type, env_cfg["environment"])

    # Prompts
    prompts_cfg = env_cfg.get("prompts", {})
    if prompt_override:
        prompts_cfg = {k: v for k, v in prompts_cfg.items() if k in prompt_override}

    prompts = {pid: pcfg["text"] for pid, pcfg in prompts_cfg.items()}

    # Find aligned/mismatched/inverted prompt IDs
    aligned_prompt = next(
        (pid for pid, pcfg in prompts_cfg.items() if pcfg.get("aligned")), None
    )
    mismatched_prompt = next(
        (pid for pid, pcfg in prompts_cfg.items() if pcfg.get("mismatched")), None
    )
    inverted_prompt = next(
        (pid for pid, pcfg in prompts_cfg.items() if pcfg.get("inverted")), None
    )

    # Trajectory counts
    traj_counts = env_cfg["environment"].get("trajectory_counts", {})
    if traj_count_override:
        traj_counts = {level: traj_count_override for level in traj_counts}

    # Cache path
    cache_path = os.path.join(output_dir, "cache", f"{exp_name}_trajectories.pkl")

    # Generate trajectories (once, shared across all models)
    logger.info("Step 1: Generating trajectories ...")
    trajectories_by_level = generate_and_score_trajectories(
        generator,
        scorer,
        quality_levels=list(traj_counts.keys()),
        counts=traj_counts,
        seed=seed,
        cache_path=cache_path if not args.skip_trajectories else None,
    )
    logger.info(
        f"Generated {sum(len(v) for v in trajectories_by_level.values())} trajectories total."
    )

    # Analysis config
    analysis_cfg = env_cfg.get("analysis", {})
    quality_metrics = analysis_cfg.get("quality_metrics", ["q_individual"])
    surprise_types = analysis_cfg.get("surprise_types", ["normalised"])
    semantic_filters = analysis_cfg.get("semantic_filters", ["all"])
    bootstrap_n = analysis_cfg.get("bootstrap_n", 1000)
    permutation_n = analysis_cfg.get("permutation_null_n", 1000)
    cf_cfg = env_cfg["environment"].get("counterfactual", {})
    n_edits_per_traj = cf_cfg.get("n_edits_per_trajectory", 2)
    n_sample_cf = cf_cfg.get("n_sample_trajectories", 100)

    # Run per-model evaluation
    all_surprise_results = []
    all_cf_results = []
    model_sizes_b: dict[str, float] = {}

    for model_cfg in models_cfg["models"]:
        if model_cfg["name"] not in model_names:
            continue

        model_name = model_cfg["name"]
        model_sizes_b[model_name] = model_cfg.get("size_b", 0.0)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Step 2: Evaluating model {model_name!r}")
        logger.info(f"{'=' * 60}")

        from ccsm_eval.evaluation.model_loader import load_model
        from ccsm_eval.evaluation.surprise import SurpriseEvaluator

        loaded_model = load_model(
            model_name=model_name,
            hf_path=model_cfg["hf_path"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            use_vllm=model_cfg.get("use_vllm", False),
        )
        evaluator = SurpriseEvaluator(
            loaded_model,
            batch_size=model_cfg.get("batch_size", 4),
            max_length=model_cfg.get("max_length", 4096),
        )

        logger.info("Step 3: Running surprise evaluation ...")
        surprise_results = run_surprise_evaluation(
            evaluator,
            trajectories_by_level,
            prompts,
            formatters,
            env_type,
        )
        all_surprise_results.extend(surprise_results)

        if not args.skip_counterfactuals:
            logger.info("Step 4: Running counterfactual edits ...")
            cf_results = run_counterfactual_edits(
                cf_editor,
                evaluator,
                trajectories_by_level,
                {k: v for k, v in prompts.items() if k == aligned_prompt or aligned_prompt is None},
                formatters,
                n_edits_per_traj=n_edits_per_traj,
                n_sample_trajectories=n_sample_cf,
                seed=seed,
            )
            all_cf_results.extend(cf_results)

        # Free GPU memory between models
        del evaluator, loaded_model
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Analysis
    logger.info("\nStep 5: Running analysis ...")
    analysis_results = run_analysis(
        all_surprise_results,
        all_cf_results,
        quality_metrics=quality_metrics,
        surprise_types=surprise_types,
        semantic_filters=semantic_filters,
        bootstrap_n=bootstrap_n,
        permutation_n=permutation_n,
        aligned_prompt=aligned_prompt,
        mismatched_prompt=mismatched_prompt,
        inverted_prompt=inverted_prompt,
    )

    # Scaling analysis (across models)
    if len(model_names) > 1:
        from ccsm_eval.analysis.scaling import analyse_scaling
        scaling_results = []
        by_model_corrs: dict[str, list] = {}
        for r in analysis_results["correlation_results"]:
            by_model_corrs.setdefault(r.model_id, []).append(r)

        for qm in quality_metrics:
            for pid in prompts.keys():
                sr = analyse_scaling(by_model_corrs, model_sizes_b, env_type, qm, pid)
                if sr:
                    scaling_results.append(sr)

        analysis_results["scaling_results"] = scaling_results

    analysis_results["surprise_results"] = all_surprise_results
    analysis_results["counterfactual_raw"] = all_cf_results

    # Save
    logger.info("\nStep 6: Saving results ...")
    save_results(analysis_results, output_dir, exp_name)

    # Figures
    logger.info("Step 7: Generating figures ...")
    try:
        from ccsm_eval.reporting.figures import save_all_figures
        figures_dir = os.path.join(output_dir, "figures")
        save_all_figures({"by_environment": {env_type: {"by_prompt": {
            r.prompt_id: {"surprise_results": [
                s for s in all_surprise_results if s.prompt_id == r.prompt_id
            ]} for r in analysis_results["correlation_results"]
        }}}}, figures_dir)
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
