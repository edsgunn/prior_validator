"""Gridworld trajectory generator.

Creates text-based navigation episodes on randomly generated grids.
Quality levels range from optimal (A* shortest path) to adversarial
(deliberately moving away from the goal).
"""

from __future__ import annotations

import collections
import heapq
import random
import uuid
from typing import Optional

from ccsm_eval.trajectories.base import Token, Trajectory


QUALITY_LEVELS = ["optimal", "near_optimal", "wandering", "lost", "adversarial"]

# Grid cell types
EMPTY = "."
WALL = "#"
START = "S"
GOAL = "G"

DIRECTIONS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


class Grid:
    """A 2D grid with walls, a start, and a goal."""

    def __init__(self, width: int, height: int, cells: list[list[str]],
                 start: tuple[int, int], goal: tuple[int, int]):
        self.width = width
        self.height = height
        self.cells = cells    # cells[y][x]
        self.start = start
        self.goal = goal

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.cells[y][x] != WALL

    def neighbors(self, x: int, y: int) -> list[tuple[str, int, int]]:
        result = []
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if self.passable(nx, ny):
                result.append((direction, nx, ny))
        return result

    def visible_directions(self, x: int, y: int) -> dict[str, str]:
        """What can the agent see from (x, y) in each direction?"""
        view = {}
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                view[direction] = "wall"
            elif self.cells[ny][nx] == WALL:
                view[direction] = "wall"
            elif (nx, ny) == self.goal:
                view[direction] = "goal"
            else:
                view[direction] = "open path"
        return view


def generate_grid(
    width: int, height: int, wall_density: float, rng: random.Random
) -> Grid:
    """Generate a random grid with walls, a start at top-left, goal at bottom-right."""
    cells = [[EMPTY for _ in range(width)] for _ in range(height)]

    start = (0, 0)
    goal = (width - 1, height - 1)

    # Place random walls (not on start/goal)
    for y in range(height):
        for x in range(width):
            if (x, y) not in (start, goal) and rng.random() < wall_density:
                cells[y][x] = WALL

    grid = Grid(width, height, cells, start, goal)

    # Ensure start-to-goal path exists via BFS; clear walls if needed
    if not _bfs_path(grid, start, goal):
        _clear_path(grid, start, goal, rng)

    return grid


def _bfs_path(
    grid: Grid, start: tuple[int, int], goal: tuple[int, int]
) -> Optional[list[str]]:
    """BFS from start to goal. Returns list of directions or None."""
    queue = collections.deque([(start, [])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for direction, nx, ny in grid.neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [direction]))
    return None


def _astar_path(
    grid: Grid, start: tuple[int, int], goal: tuple[int, int]
) -> Optional[list[str]]:
    """A* from start to goal. Returns list of directions or None."""
    gx, gy = goal

    def h(x: int, y: int) -> int:
        return abs(x - gx) + abs(y - gy)

    heap = [(h(*start), 0, start, [])]
    visited: dict[tuple, int] = {}

    while heap:
        f, g, (x, y), path = heapq.heappop(heap)
        if (x, y) == goal:
            return path
        if visited.get((x, y), float("inf")) <= g:
            continue
        visited[(x, y)] = g
        for direction, nx, ny in grid.neighbors(x, y):
            ng = g + 1
            if visited.get((nx, ny), float("inf")) > ng:
                heapq.heappush(heap, (ng + h(nx, ny), ng, (nx, ny), path + [direction]))
    return None


def _clear_path(
    grid: Grid, start: tuple[int, int], goal: tuple[int, int], rng: random.Random
) -> None:
    """Remove walls along a random path from start to goal to guarantee connectivity."""
    x, y = start
    gx, gy = goal
    while (x, y) != (gx, gy):
        if x < gx and rng.random() < 0.5:
            x += 1
        elif y < gy and rng.random() < 0.5:
            y += 1
        elif x < gx:
            x += 1
        else:
            y += 1
        grid.cells[y][x] = EMPTY


class GridworldTrajectoryGenerator:
    """Generates navigation trajectories at controlled quality levels.

    Args:
        min_size: Minimum grid side length.
        max_size: Maximum grid side length.
        wall_density: Fraction of non-start/goal cells that are walls.
        max_steps: Maximum steps before episode ends (failure).
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 8,
        wall_density: float = 0.2,
        max_steps: int = 100,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.wall_density = wall_density
        self.max_steps = max_steps

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(
        self, quality_level: str, n_trajectories: int, seed: int
    ) -> list[Trajectory]:
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        rng = random.Random(seed)
        trajectories = []
        for _ in range(n_trajectories):
            w = rng.randint(self.min_size, self.max_size)
            h = rng.randint(self.min_size, self.max_size)
            grid = generate_grid(w, h, self.wall_density, rng)
            traj_seed = rng.randint(0, 2**31)
            traj = self._generate_one(grid, quality_level, random.Random(traj_seed))
            trajectories.append(traj)
        return trajectories

    def _generate_one(
        self, grid: Grid, quality_level: str, rng: random.Random
    ) -> Trajectory:
        optimal_path = _astar_path(grid, grid.start, grid.goal) or []
        optimal_length = len(optimal_path)

        steps = self._execute_policy(grid, quality_level, optimal_path, rng)
        tokens = self._build_tokens(grid, steps)

        reached_goal = steps and steps[-1]["reached_goal"]
        actual_length = len(steps)

        quality = (optimal_length / actual_length) if (reached_goal and actual_length > 0) else 0.0

        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            tokens=tokens,
            character_prompt="",
            quality_scores={},
            quality_level=quality_level,
            environment="gridworld",
            metadata={
                "grid_width": grid.width,
                "grid_height": grid.height,
                "start": grid.start,
                "goal": grid.goal,
                "optimal_path": optimal_path,
                "optimal_length": optimal_length,
                "actual_length": actual_length,
                "reached_goal": reached_goal,
                "path_optimality": quality,
                "steps": [
                    {"direction": s["direction"], "x": s["x"], "y": s["y"]}
                    for s in steps
                ],
            },
        )

    def _execute_policy(
        self,
        grid: Grid,
        quality_level: str,
        optimal_path: list[str],
        rng: random.Random,
    ) -> list[dict]:
        x, y = grid.start
        gx, gy = grid.goal
        steps: list[dict] = []

        for step_i in range(self.max_steps):
            if (x, y) == (gx, gy):
                break

            direction = self._choose_direction(
                grid, x, y, gx, gy, quality_level, optimal_path, step_i, rng
            )
            if direction is None:
                break  # stuck

            dx, dy = DIRECTIONS[direction]
            nx, ny = x + dx, y + dy

            if not grid.passable(nx, ny):
                # Bounce — stay in place, record bump
                steps.append({
                    "direction": direction,
                    "x": x, "y": y,
                    "bumped": True,
                    "reached_goal": False,
                })
                continue

            x, y = nx, ny
            steps.append({
                "direction": direction,
                "x": x, "y": y,
                "bumped": False,
                "reached_goal": (x, y) == (gx, gy),
            })

        return steps

    def _choose_direction(
        self,
        grid: Grid,
        x: int, y: int,
        gx: int, gy: int,
        quality_level: str,
        optimal_path: list[str],
        step_i: int,
        rng: random.Random,
    ) -> Optional[str]:
        neighbors = grid.neighbors(x, y)
        if not neighbors:
            return None

        valid_dirs = {d for d, _, _ in neighbors}

        if quality_level == "optimal":
            if step_i < len(optimal_path):
                d = optimal_path[step_i]
                return d if d in valid_dirs else (rng.choice(list(valid_dirs)) if valid_dirs else None)
            return None

        elif quality_level == "near_optimal":
            # Follow optimal path with occasional 1-2 step detour
            if step_i < len(optimal_path) and rng.random() < 0.85:
                d = optimal_path[step_i]
                return d if d in valid_dirs else rng.choice(list(valid_dirs))
            return rng.choice(list(valid_dirs))

        elif quality_level == "wandering":
            # Biased random walk toward goal
            goal_dirs: list[str] = []
            if x < gx:
                goal_dirs.append("east")
            if x > gx:
                goal_dirs.append("west")
            if y < gy:
                goal_dirs.append("south")
            if y > gy:
                goal_dirs.append("north")
            goal_dirs = [d for d in goal_dirs if d in valid_dirs]
            if goal_dirs and rng.random() < 0.6:
                return rng.choice(goal_dirs)
            return rng.choice(list(valid_dirs))

        elif quality_level == "lost":
            # Uniform random walk
            return rng.choice(list(valid_dirs))

        elif quality_level == "adversarial":
            # Move away from goal
            away_dirs: list[str] = []
            if x > gx:
                away_dirs.append("east")
            if x < gx:
                away_dirs.append("west")
            if y > gy:
                away_dirs.append("south")
            if y < gy:
                away_dirs.append("north")
            away_dirs = [d for d in away_dirs if d in valid_dirs]
            if away_dirs and rng.random() < 0.8:
                return rng.choice(away_dirs)
            return rng.choice(list(valid_dirs))

        return rng.choice(list(valid_dirs))

    @staticmethod
    def _build_tokens(grid: Grid, steps: list[dict]) -> list[Token]:
        tokens: list[Token] = []
        position = 0

        # Initial observation
        sx, sy = grid.start
        gx, gy = grid.goal
        initial_obs = (
            f"You are at position ({sx + 1},{sy + 1}) in a "
            f"{grid.width}x{grid.height} grid. "
            f"The goal is at ({gx + 1},{gy + 1})."
        )
        vis = grid.visible_directions(sx, sy)
        vis_str = ", ".join(f"{d}: {v}" for d, v in vis.items())
        initial_obs += f" You can see: {vis_str}."

        tokens.append(Token(
            text=initial_obs,
            token_ids=[],
            is_observation=True,
            semantic_type="position_description",
            position=position,
        ))
        position += 1

        cx, cy = grid.start
        for step in steps:
            direction = step["direction"]

            # Action token
            tokens.append(Token(
                text=f"move {direction}",
                token_ids=[],
                is_observation=False,
                semantic_type="movement_action",
                position=position,
            ))
            position += 1

            # Observation: result of move
            nx, ny = step["x"], step["y"]
            if step.get("bumped"):
                obs_text = f"You tried to move {direction} but hit a wall. You remain at ({cx + 1},{cy + 1})."
            elif step.get("reached_goal"):
                obs_text = f"You moved {direction}. You reached the goal at ({nx + 1},{ny + 1})!"
                cx, cy = nx, ny
            else:
                cx, cy = nx, ny
                vis = grid.visible_directions(cx, cy)
                vis_str = ", ".join(f"{d}: {v}" for d, v in vis.items())
                obs_text = (
                    f"You moved {direction}. You are now at ({cx + 1},{cy + 1}). "
                    f"You can see: {vis_str}."
                )

            tokens.append(Token(
                text=obs_text,
                token_ids=[],
                is_observation=True,
                semantic_type="position_description",
                position=position,
            ))
            position += 1

        return tokens
