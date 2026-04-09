"""Shared abstract base classes for Phase 2 live environments and counterfactual editors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextEnvironment(ABC):
    """Phase 2: Live text-based environment for LLM agent interaction.

    Follows a minimal PettingZoo-style turn-based API.  Only one agent acts at
    a time; scripted NPCs are stepped internally by the environment.
    """

    @abstractmethod
    def reset(self) -> dict[str, str]:
        """Reset and return initial observations.

        Returns:
            Mapping of agent_id -> observation string for each agent that
            receives an observation at episode start (typically just "agent").
        """

    @abstractmethod
    def step(self, agent_id: str, action: str) -> dict:
        """Process the action and return observations + terminal info.

        Returns a dict containing:
            - One key per agent_id that receives an observation (str values).
            - "done" (bool): whether the episode has ended.
            - "info" (dict): episode statistics (scores, metadata, etc.).
        """

    @abstractmethod
    def current_agent(self) -> str:
        """Return the agent_id that should act next."""


class CounterfactualEditor(ABC):
    """Phase 1: Produce counterfactual trajectory edits for causal testing."""

    @abstractmethod
    def sample_edit_positions(self, trajectory) -> list[int]:
        """Return token positions eligible for editing (action tokens only)."""

    @abstractmethod
    def edit(self, trajectory, position: int, direction: str):
        """Replace the action at *position* and propagate consequences.

        Args:
            trajectory: The original Trajectory.
            position: Index into trajectory.tokens of the action to replace.
            direction: "up" for a better action, "down" for a worse one.

        Returns:
            A CounterfactualEdit dataclass (from ccsm_eval.trajectories.base).
        """
