"""Gridworld trajectory formatter.

Converts a Trajectory produced by GridworldTrajectoryGenerator into a
text string with character-level observation mask and semantic type map.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


def _append(
    text_parts: list[str],
    mask_parts: list[list[bool]],
    type_parts: list[list[str]],
    chunk: str,
    is_obs: bool,
    sem_type: str,
) -> None:
    text_parts.append(chunk)
    mask_parts.append([is_obs] * len(chunk))
    type_parts.append([sem_type] * len(chunk))


def _assemble(
    text_parts: list[str],
    mask_parts: list[list[bool]],
    type_parts: list[list[str]],
) -> tuple[str, list[bool], list[str]]:
    full_text = "".join(text_parts)
    obs_mask: list[bool] = []
    sem_types: list[str] = []
    for m, t in zip(mask_parts, type_parts):
        obs_mask.extend(m)
        sem_types.extend(t)
    return full_text, obs_mask, sem_types


class GridworldFormatter:
    """Formats gridworld trajectories as natural language text.

    Format:
        <prompt>\\n
        <initial position description>\\n
        > move east\\n
        <resulting position description>\\n
        > move south\\n
        ...
    """

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        prompt_block = character_prompt.strip() + "\n"
        _append(text_parts, mask_parts, type_parts, prompt_block, False, "prompt")

        tokens = trajectory.tokens
        if not tokens:
            return _assemble(text_parts, mask_parts, type_parts)

        for tok in tokens:
            if tok.is_observation:
                chunk = tok.text.strip() + "\n"
                _append(text_parts, mask_parts, type_parts, chunk, True, tok.semantic_type)
            else:
                chunk = f"> {tok.text.strip()}\n"
                _append(text_parts, mask_parts, type_parts, chunk, False, tok.semantic_type)

        # Append outcome summary
        reached = trajectory.metadata.get("reached_goal", False)
        optimality = trajectory.metadata.get("path_optimality", 0.0)
        if reached:
            summary = f"[Navigation complete. Path efficiency: {optimality:.0%}]\n"
        else:
            summary = "[Navigation failed: goal not reached within step limit.]\n"
        _append(text_parts, mask_parts, type_parts, summary, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)
