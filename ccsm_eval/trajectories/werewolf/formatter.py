"""Format Werewolf trajectories as text with character-level observation masks."""

from __future__ import annotations

from ccsm_eval.trajectories.base import Token, Trajectory


class WerewolfFormatter:
    """Formats a Werewolf trajectory into a flat text string with per-character masks."""

    def format(
        self, trajectory: Trajectory, prompt_text: str
    ) -> tuple[str, list[bool], list[str]]:
        """Format trajectory as text with observation mask and semantic type lists.

        Args:
            trajectory: The Werewolf trajectory to format.
            prompt_text: Character/system prompt to prepend.

        Returns:
            full_text: Complete formatted string.
            obs_mask: Per-character bool — True if character is observation text.
            sem_types: Per-character semantic type string.
        """
        parts: list[str] = []
        obs_mask: list[bool] = []
        sem_types: list[str] = []

        # Prepend prompt (not an observation)
        prompt_section = prompt_text + "\n\n"
        parts.append(prompt_section)
        obs_mask.extend([False] * len(prompt_section))
        sem_types.extend(["prompt"] * len(prompt_section))

        for token in trajectory.tokens:
            if token.is_observation:
                formatted = f"[Observation: {token.text}]\n"
                is_obs = True
            else:
                formatted = f"[Action: {token.text}]\n"
                is_obs = False

            parts.append(formatted)
            obs_mask.extend([is_obs] * len(formatted))
            sem_types.extend([token.semantic_type] * len(formatted))

        full_text = "".join(parts)
        return full_text, obs_mask, sem_types
