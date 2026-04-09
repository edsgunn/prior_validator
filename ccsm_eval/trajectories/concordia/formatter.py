"""Format Concordia trajectories as text with character-level observation masks.

Follows the same interface as chess/negotiation/gridworld formatters:
    format(trajectory, prompt_text) -> (full_text, obs_mask, sem_types)

Observation tokens → "[Observation: {text}]\n"
Action tokens      → "[Action: {text}]\n"
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


class ConcordiaFormatter:
    """Formats Concordia trajectories for surprise evaluation."""

    def format(
        self, trajectory: Trajectory, prompt_text: str
    ) -> tuple[str, list[bool], list[str]]:
        """Format trajectory as text with character-level masks.

        Args:
            trajectory: The trajectory to format.
            prompt_text: Character prompt prepended to the full text.

        Returns:
            (full_text, obs_mask, sem_types) where each list has one entry
            per character in full_text.
        """
        parts: list[str] = []
        masks: list[bool] = []
        types: list[str] = []

        def append(text: str, is_obs: bool, sem_type: str) -> None:
            parts.append(text)
            masks.extend([is_obs] * len(text))
            types.extend([sem_type] * len(text))

        # Prompt is not an observation
        prompt_chunk = prompt_text + "\n\n"
        append(prompt_chunk, False, "prompt")

        for token in trajectory.tokens:
            if token.is_observation:
                chunk = f"[Observation: {token.text}]\n"
                append(chunk, True, token.semantic_type)
            else:
                chunk = f"[Action: {token.text}]\n"
                append(chunk, False, token.semantic_type)

        return "".join(parts), masks, types
