"""Core dataclasses for the CCSM Phase 1 evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Token:
    """A single token in a trajectory.

    The is_observation flag (σ_t) distinguishes observation tokens (produced by
    the environment) from action tokens (produced by the agent). Surprise is
    only accumulated over observation tokens.
    """

    text: str
    token_ids: list[int]          # After tokenisation (model-specific)
    is_observation: bool           # σ_t: True for observation, False for action
    semantic_type: str             # e.g. "opponent_move", "board_state", "offer",
                                   #      "narration" — for stratified analysis
    position: int                  # Position in the full trajectory


@dataclass
class Trajectory:
    """A complete trajectory with tokens, quality scores, and metadata."""

    trajectory_id: str
    tokens: list[Token]
    character_prompt: str
    quality_scores: dict[str, float]   # e.g. {"centipawn": 0.85, "outcome": 1.0}
    quality_level: str                 # e.g. "optimal", "weak", "random"
    environment: str                   # e.g. "chess", "negotiation", "gridworld"
    metadata: dict = field(default_factory=dict)


@dataclass
class CounterfactualEdit:
    """A counterfactual edit replacing one action with a better or worse one.

    The edit replaces a single action token at `edit_position` and propagates
    the consequences: the environment responds to the new action, producing
    a new observation sequence stored in `replacement_tokens`.
    """

    trajectory_id: str
    edit_position: int                  # Token position of the replaced action
    original_action: str
    replacement_action: str
    direction: str                      # "up" (better) or "down" (worse)
    quality_delta: float                # Δ quality (positive = improvement)
    original_tokens: list[Token]        # Observation tokens after original action
    replacement_tokens: list[Token]     # Observation tokens after replacement action


@dataclass
class SurpriseResult:
    """The output of a surprise evaluation for a single trajectory + prompt."""

    trajectory_id: str
    prompt_id: str
    model_id: str
    per_token_surprise: list[float]           # s_t for each observation token
    per_token_semantic_type: list[str]        # Semantic type of each obs token
    per_token_unigram_logprob: list[float]    # For residualisation
    cumulative_surprise: float                # S(x_{1:T}, c)
    normalised_surprise: float                # S / N_obs
    quality_scores: dict[str, float]
    quality_level: str
    environment: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CounterfactualSurpriseResult:
    """Surprise measurements on the observation tokens after a counterfactual edit."""

    trajectory_id: str
    edit_position: int
    direction: str                  # "up" or "down"
    quality_delta: float
    surprise_original: float        # Surprise over obs tokens after original action
    surprise_replacement: float     # Surprise over obs tokens after replacement action
    delta_surprise: float           # surprise_replacement - surprise_original
    prompt_id: str
    model_id: str


@dataclass
class TrajectoryBatch:
    """A batch of trajectories for efficient parallel processing."""

    trajectories: list[Trajectory]
    environment: str
    quality_level: Optional[str] = None   # Set if all trajectories share a level

    def __len__(self) -> int:
        return len(self.trajectories)
