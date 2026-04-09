"""CLI script: generate Concordia trajectories and save to disk.

Usage:
    uv run python -m ccsm_eval.trajectories.concordia.run_generation \\
        --scenarios resource_division public_goods stag_hunt \\
        --quality-levels expert moderate poor random \\
        --n 5 \\
        --output data/concordia_trajectories \\
        --model claude-sonnet-4-20250514

This script is for offline trajectory generation. Run it once; the resulting
trajectories are loaded by run_eval.py during the main evaluation pipeline.

For a quick end-to-end test, use --n 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("concordia.run_generation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Concordia trajectories for CCSM Phase 1.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["resource_division", "public_goods", "stag_hunt"],
        help="Scenarios to run.",
    )
    parser.add_argument(
        "--quality-levels",
        nargs="+",
        default=["expert", "moderate", "poor", "random"],
        help="Quality levels for the focal agent.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of trajectories per (scenario, quality_level) combination.",
    )
    parser.add_argument(
        "--output",
        default="data/concordia_trajectories",
        help="Root output directory for trajectories.",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="API provider for agents and GM.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model ID for agents and GM. "
            "Defaults: anthropic=claude-sonnet-4-20250514, openai=gpt-4o."
        ),
    )
    parser.add_argument(
        "--npc-quality",
        default="moderate",
        help="Quality level for non-focal (NPC) agents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Falls back to ANTHROPIC_API_KEY or OPENAI_API_KEY env var.",
    )
    args = parser.parse_args()

    provider = args.provider
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = args.api_key or os.environ.get(env_var, "")
    if not api_key:
        logger.error(
            f"No API key for {provider}. Set {env_var} or pass --api-key."
        )
        sys.exit(1)

    from ccsm_eval.trajectories.concordia.generator import ConcordiaTrajectoryGenerator

    output_root = Path(args.output)
    manifest_rows: list[dict] = []

    for scenario_name in args.scenarios:
        logger.info(f"=== Scenario: {scenario_name} ===")
        generator = ConcordiaTrajectoryGenerator(
            scenario_name=scenario_name,
            provider=args.provider,
            model=args.model,  # None → make_language_model picks default for provider
            api_key=api_key,
            npc_quality_level=args.npc_quality,
            save_dir=str(output_root),
        )

        for quality_level in args.quality_levels:
            logger.info(f"  Quality level: {quality_level} (n={args.n})")
            traj_dir = output_root / scenario_name / quality_level
            traj_dir.mkdir(parents=True, exist_ok=True)

            import random
            level_seed = args.seed + hash(f"{scenario_name}_{quality_level}") % (2**20)
            trajectories = generator.generate(quality_level, n=args.n, seed=level_seed)

            for i, traj in enumerate(trajectories):
                # Save as JSON
                json_path = traj_dir / f"trajectory_{i:03d}_{traj.trajectory_id[:8]}.json"
                payload = {
                    "trajectory_id": traj.trajectory_id,
                    "scenario_name": scenario_name,
                    "quality_level": quality_level,
                    "quality_scores": traj.quality_scores,
                    "metadata": traj.metadata,
                    "tokens": [
                        {
                            "text": t.text,
                            "is_observation": t.is_observation,
                            "semantic_type": t.semantic_type,
                            "position": t.position,
                        }
                        for t in traj.tokens
                    ],
                }
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2)

                # Also save as pickle for fast loading in run_eval.py
                pkl_path = traj_dir / f"trajectory_{i:03d}_{traj.trajectory_id[:8]}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(traj, f)

                manifest_rows.append({
                    "trajectory_id": traj.trajectory_id,
                    "scenario": scenario_name,
                    "quality_level": quality_level,
                    "focal_agent": traj.metadata.get("focal_agent_name", "Agent_A"),
                    "payoff": traj.quality_scores.get("payoff", 0.0),
                    "social_welfare": traj.quality_scores.get("social_welfare", 0.0),
                    "cooperation_score": traj.quality_scores.get("cooperation_score", 0.0),
                    "n_tokens": len(traj.tokens),
                    "n_turns": traj.metadata.get("n_turns", 0),
                    "filepath": str(pkl_path),
                })

                logger.info(
                    f"    [{i+1}/{args.n}] payoff={traj.quality_scores.get('payoff', 0):.2f} "
                    f"sw={traj.quality_scores.get('social_welfare', 0):.2f} "
                    f"tokens={len(traj.tokens)}"
                )

    # Write manifest CSV
    manifest_path = output_root / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        logger.info(f"Manifest written to {manifest_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Scenario':<25} {'Quality':<12} {'N':>4} {'Mean payoff':>12} {'Mean sw':>10} {'Mean coop':>11}")
    print("-" * 78)
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for row in manifest_rows:
        groups[(row["scenario"], row["quality_level"])].append(row)
    for (scenario, ql), rows in sorted(groups.items()):
        n = len(rows)
        mean_pay = sum(r["payoff"] for r in rows) / n
        mean_sw = sum(r["social_welfare"] for r in rows) / n
        mean_coop = sum(r["cooperation_score"] for r in rows) / n
        print(f"{scenario:<25} {ql:<12} {n:>4} {mean_pay:>12.3f} {mean_sw:>10.3f} {mean_coop:>11.3f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
