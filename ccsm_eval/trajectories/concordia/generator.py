"""ConcordiaTrajectoryGenerator — main Phase 1 trajectory generator.

Wraps the three social interaction scenarios (ResourceDivisionScenario,
PublicGoodsScenario, StaggeredCoordinationScenario) with quality-level
control via ConcordiaAgent personas and LLM temperature.

Usage in run_eval.py:
    generator = ConcordiaTrajectoryGenerator(
        scenario_name="resource_division",
        anthropic_model="claude-sonnet-4-20250514",
        api_key=...,
    )
    trajectories = generator.generate("expert", n=5, seed=42)

Note on Concordia:
    The actual Concordia library (pip install gdm-concordia) requires Python >=3.12.
    This generator implements an equivalent pipeline using the Anthropic API directly
    and is designed to be API-compatible with a future Concordia-backed version.
    When the project upgrades to Python 3.12, replace AnthropicLanguageModel with
    a native Concordia LanguageModel adapter and the scenario classes with
    Concordia prefab environments.
"""

from __future__ import annotations

import json
import logging
import os
import random
import uuid
from pathlib import Path
from typing import Optional

from ccsm_eval.trajectories.base import Token, Trajectory
from ccsm_eval.trajectories.concordia.agent_factories import (
    QUALITY_LEVELS,
    make_agent,
    make_game_master,
)
from ccsm_eval.trajectories.concordia.language_model import make_language_model
from ccsm_eval.trajectories.concordia.scenarios import (
    EpisodeLog,
    PublicGoodsScenario,
    ResourceDivisionScenario,
    StaggeredCoordinationScenario,
)
from ccsm_eval.trajectories.concordia.transcript_parser import (
    episode_to_trajectory,
    validate_trajectory,
)

logger = logging.getLogger(__name__)

SCENARIO_NAMES = ["resource_division", "public_goods", "stag_hunt"]


class ConcordiaTrajectoryGenerator:
    """Generates Concordia-style social interaction trajectories at controlled quality levels.

    Args:
        scenario_name: Which scenario to run. One of "resource_division",
            "public_goods", "stag_hunt".
        model: Model ID (e.g. "claude-sonnet-4-20250514" or "gpt-4o").
        provider: API provider — "anthropic" (default) or "openai".
        api_key: API key. Falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY env vars.
        npc_quality_level: Quality level for non-focal agents (default "moderate").
        save_dir: If provided, save raw episode logs as JSON alongside trajectories.
        model_kwargs: Extra kwargs forwarded to the language model constructor
                      (e.g. base_url for OpenAI-compatible endpoints).
    """

    def __init__(
        self,
        scenario_name: str = "resource_division",
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        npc_quality_level: str = "moderate",
        save_dir: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        if scenario_name not in SCENARIO_NAMES:
            raise ValueError(
                f"Unknown scenario {scenario_name!r}. Must be one of {SCENARIO_NAMES}"
            )
        self.scenario_name = scenario_name
        self.npc_quality_level = npc_quality_level
        self.save_dir = Path(save_dir) if save_dir else None

        self._model = make_language_model(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.7,  # overridden per agent
            **(model_kwargs or {}),
        )

    def quality_levels(self) -> list[str]:
        return list(QUALITY_LEVELS)

    def generate(
        self, quality_level: str, n: int, seed: int = 42
    ) -> list[Trajectory]:
        """Generate n trajectories at the given quality level.

        Args:
            quality_level: Focal agent quality. One of "expert", "moderate", "poor", "random".
            n: Number of trajectories to generate.
            seed: Random seed for reproducibility (affects scenario parameters
                  and agent history; LLM sampling is not fully deterministic).

        Returns:
            List of Trajectory objects with quality_scores populated.
        """
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(
                f"Unknown quality level {quality_level!r}. Must be one of {QUALITY_LEVELS}"
            )

        rng = random.Random(seed)
        trajectories: list[Trajectory] = []

        for i in range(n):
            traj_seed = rng.randint(0, 2**31)
            try:
                traj = self._generate_one(quality_level, seed=traj_seed, index=i)
                trajectories.append(traj)
                logger.debug(
                    f"[{self.scenario_name}/{quality_level}] traj {i+1}/{n}: "
                    f"payoff={traj.quality_scores.get('payoff', '?'):.2f}, "
                    f"tokens={len(traj.tokens)}"
                )
            except Exception as exc:
                logger.error(
                    f"[{self.scenario_name}/{quality_level}] traj {i+1}/{n} failed: {exc}"
                )
                raise

        return trajectories

    def _generate_one(
        self, quality_level: str, seed: int, index: int
    ) -> Trajectory:
        rng = random.Random(seed)

        focal_agent = make_agent(
            name="Agent_A",
            quality_level=quality_level,
            scenario_name=self.scenario_name,
            model=self._model,
        )
        game_master = make_game_master(self.scenario_name, self._model)

        if self.scenario_name == "resource_division":
            scenario = ResourceDivisionScenario(max_rounds=8)
            npc = make_agent(
                name="Agent_B",
                quality_level=self.npc_quality_level,
                scenario_name=self.scenario_name,
                model=self._model,
            )
            scenario_data = scenario.generate_scenario(rng)
            log = scenario.run(
                focal_agent=focal_agent,
                other_agent=npc,
                game_master=game_master,
                scenario_data=scenario_data,
                rng=rng,
                focal_is_A=True,
            )

        elif self.scenario_name == "public_goods":
            scenario = PublicGoodsScenario(n_rounds=5)
            npcs = [
                make_agent(
                    name=f"Agent_{chr(66+i)}",
                    quality_level=self.npc_quality_level,
                    scenario_name=self.scenario_name,
                    model=self._model,
                )
                for i in range(2)
            ]
            log = scenario.run(
                focal_agent=focal_agent,
                other_agents=npcs,
                game_master=game_master,
                rng=rng,
            )

        elif self.scenario_name == "stag_hunt":
            scenario = StaggeredCoordinationScenario(discussion_rounds=4)
            npc = make_agent(
                name="Agent_B",
                quality_level=self.npc_quality_level,
                scenario_name=self.scenario_name,
                model=self._model,
            )
            log = scenario.run(
                focal_agent=focal_agent,
                other_agent=npc,
                game_master=game_master,
                rng=rng,
            )

        else:
            raise ValueError(f"Unhandled scenario: {self.scenario_name}")

        log.quality_level = quality_level
        traj = episode_to_trajectory(log, quality_level=quality_level)

        # Validate and log warnings
        warnings = validate_trajectory(traj)
        for w in warnings:
            logger.warning(f"[{self.scenario_name}/{quality_level}/{index}] {w}")

        # Optionally persist raw episode log
        if self.save_dir:
            self._save_log(log, quality_level, index, traj.trajectory_id)

        return traj

    def _save_log(
        self,
        log: EpisodeLog,
        quality_level: str,
        index: int,
        trajectory_id: str,
    ) -> None:
        out_dir = self.save_dir / self.scenario_name / quality_level
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"trajectory_{index:03d}_{trajectory_id[:8]}.json"
        payload = {
            "trajectory_id": trajectory_id,
            "scenario_name": log.scenario_name,
            "focal_agent_name": log.focal_agent_name,
            "quality_level": quality_level,
            "quality_scores": log.quality_scores,
            "metadata": log.metadata,
            "entries": [
                {
                    "turn": e.turn,
                    "speaker": e.speaker,
                    "text": e.text,
                    "semantic_type": e.semantic_type,
                    "is_focal_action": e.is_focal_action,
                }
                for e in log.entries
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
