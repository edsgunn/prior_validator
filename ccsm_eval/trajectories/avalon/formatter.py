"""Avalon trajectory formatter: converts Trajectory to character-level text + masks."""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory, Token


class AvalonFormatter:
    """Format an Avalon Trajectory into a single text string with observation masks.

    The format matches the Werewolf formatter pattern:
      - Observations are wrapped: "[Observation: {text}]\\n"
      - Actions are wrapped:      "[Action: {text}]\\n"

    Returns a tuple of (full_text, obs_mask, sem_types) where:
      - full_text: complete string with prompt + all tokens formatted
      - obs_mask: per-character bool list (True where text is from an observation token)
      - sem_types: per-character semantic type strings
    """

    def format(
        self, trajectory: Trajectory, prompt_text: str
    ) -> tuple[str, list[bool], list[str]]:
        """Format the trajectory into text with character-level observation masks.

        Args:
            trajectory: The Avalon Trajectory to format.
            prompt_text: Character/persona prompt prepended before the trajectory.

        Returns:
            (full_text, obs_mask, sem_types):
                full_text  — complete formatted string
                obs_mask   — list[bool], one entry per character in full_text
                sem_types  — list[str], one entry per character in full_text
        """
        parts: list[str] = []
        is_obs_flags: list[bool] = []
        sem_type_flags: list[str] = []

        # --- Prepend prompt ---
        prompt_block = prompt_text + "\n\n"
        parts.append(prompt_block)
        is_obs_flags.extend([False] * len(prompt_block))
        sem_type_flags.extend(["prompt"] * len(prompt_block))

        # --- Format each token ---
        for token in trajectory.tokens:
            if token.is_observation:
                formatted = f"[Observation: {token.text}]\n"
                is_obs = True
            else:
                formatted = f"[Action: {token.text}]\n"
                is_obs = False

            parts.append(formatted)
            is_obs_flags.extend([is_obs] * len(formatted))
            sem_type_flags.extend([token.semantic_type] * len(formatted))

        full_text = "".join(parts)
        return full_text, is_obs_flags, sem_type_flags
