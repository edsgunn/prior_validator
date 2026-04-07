"""Gridworld quality scorer.

The primary metric is path optimality: Q = L_optimal / L_actual.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


class GridworldQualityScorer:
    """Scores gridworld trajectories by path optimality."""

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        meta = trajectory.metadata
        optimal_length: int = meta.get("optimal_length", 0)
        actual_length: int = meta.get("actual_length", 0)
        reached_goal: bool = meta.get("reached_goal", False)

        if not reached_goal or actual_length == 0:
            path_optimality = 0.0
        elif optimal_length == 0:
            path_optimality = 1.0  # trivial grid, start == goal
        else:
            path_optimality = optimal_length / actual_length

        return {
            "path_optimality": path_optimality,
            "reached_goal": float(reached_goal),
            "step_count": float(actual_length),
        }
