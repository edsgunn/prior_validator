"""Counterfactual editor for gridworld trajectories.

Replaces a single movement action with a better or worse alternative
and propagates the result one step forward.
"""

from __future__ import annotations

import random
from typing import Optional

from ccsm_eval.trajectories.base import CounterfactualEdit, Token, Trajectory
from ccsm_eval.trajectories.gridworld.generator import (
    DIRECTIONS,
    Grid,
    _astar_path,
    generate_grid,
)


class GridworldCounterfactualEditor:
    """Replaces one movement step with a better/worse alternative.

    "up"   = a move that reduces distance to goal (or the optimal next move).
    "down" = a move that increases distance to goal (or random non-optimal).
    """

    def sample_edit_positions(
        self, trajectory: Trajectory, n_edits: int, seed: int
    ) -> list[int]:
        """Return token indices for movement action tokens."""
        tokens = trajectory.tokens
        action_indices = [
            i for i, t in enumerate(tokens)
            if not t.is_observation and t.semantic_type == "movement_action"
        ]
        # Skip first and last
        candidates = action_indices[1:-1] if len(action_indices) > 2 else action_indices
        rng = random.Random(seed)
        n = min(n_edits, len(candidates))
        return rng.sample(candidates, n) if n > 0 else []

    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        meta = trajectory.metadata
        grid_width: int = meta["grid_width"]
        grid_height: int = meta["grid_height"]
        goal: tuple[int, int] = tuple(meta["goal"])
        steps: list[dict] = meta.get("steps", [])

        # Find which step index this token position corresponds to
        action_token_positions = [
            i for i, t in enumerate(trajectory.tokens)
            if not t.is_observation and t.semantic_type == "movement_action"
        ]
        step_index = action_token_positions.index(position) if position in action_token_positions else 0

        # Determine agent position just before this step
        if step_index == 0:
            cx, cy = meta["start"]
        else:
            prev_step = steps[step_index - 1]
            cx, cy = prev_step["x"], prev_step["y"]

        original_direction = steps[step_index]["direction"] if step_index < len(steps) else "east"

        # Reconstruct grid (without stored wall data, use a placeholder — in practice
        # the grid should be stored in metadata; here we work from steps only)
        replacement_direction, q_delta = self._choose_replacement(
            cx, cy, goal, original_direction, direction
        )

        # Build observation tokens for original move
        original_obs = self._step_observation(
            cx, cy, original_direction, goal, grid_width, grid_height
        )
        replacement_obs = self._step_observation(
            cx, cy, replacement_direction, goal, grid_width, grid_height
        )

        return CounterfactualEdit(
            trajectory_id=trajectory.trajectory_id,
            edit_position=position,
            original_action=f"move {original_direction}",
            replacement_action=f"move {replacement_direction}",
            direction=direction,
            quality_delta=q_delta,
            original_tokens=original_obs,
            replacement_tokens=replacement_obs,
        )

    def _choose_replacement(
        self,
        cx: int, cy: int,
        goal: tuple[int, int],
        original_direction: str,
        edit_direction: str,
    ) -> tuple[str, float]:
        gx, gy = goal
        orig_dx, orig_dy = DIRECTIONS[original_direction]
        orig_dist = abs(cx + orig_dx - gx) + abs(cy + orig_dy - gy)
        current_dist = abs(cx - gx) + abs(cy - gy)

        if edit_direction == "up":
            # Choose the direction that minimises Manhattan distance
            best_dir = original_direction
            best_dist = orig_dist
            for d, (dx, dy) in DIRECTIONS.items():
                if d == original_direction:
                    continue
                new_dist = abs(cx + dx - gx) + abs(cy + dy - gy)
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_dir = d
            q_delta = (orig_dist - best_dist) / max(1, current_dist)
        else:
            # Choose the direction that maximises Manhattan distance
            worst_dir = original_direction
            worst_dist = orig_dist
            for d, (dx, dy) in DIRECTIONS.items():
                if d == original_direction:
                    continue
                new_dist = abs(cx + dx - gx) + abs(cy + dy - gy)
                if new_dist > worst_dist:
                    worst_dist = new_dist
                    worst_dir = d
            best_dir = worst_dir
            q_delta = (orig_dist - worst_dist) / max(1, current_dist)  # negative

        return best_dir, q_delta

    @staticmethod
    def _step_observation(
        cx: int, cy: int,
        direction: str,
        goal: tuple[int, int],
        grid_width: int,
        grid_height: int,
    ) -> list[Token]:
        dx, dy = DIRECTIONS[direction]
        nx, ny = cx + dx, cy + dy
        gx, gy = goal

        # Boundary check (simplified — no wall info)
        if 0 <= nx < grid_width and 0 <= ny < grid_height:
            if (nx, ny) == (gx, gy):
                obs_text = f"You moved {direction}. You reached the goal at ({nx + 1},{ny + 1})!"
            else:
                obs_text = f"You moved {direction}. You are now at ({nx + 1},{ny + 1})."
        else:
            obs_text = f"You tried to move {direction} but hit the boundary. You remain at ({cx + 1},{cy + 1})."

        return [
            Token(
                text=obs_text,
                token_ids=[],
                is_observation=True,
                semantic_type="position_description",
                position=0,
            )
        ]
