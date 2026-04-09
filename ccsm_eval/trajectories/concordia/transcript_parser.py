"""Convert Concordia EpisodeLogs into our standard Token/Trajectory format.

Token classification rules (from spec):
    focal agent speaks → is_observation=False  (action token)
    everything else    → is_observation=True   (observation token)

Semantic types:
    gm_narration          — GM describing scene, resolving actions
    other_agent_statement — Another agent's dialogue
    outcome               — Round results, scores, consequences
    setup                 — Initial scenario description
    game_event            — Environmental events not caused by any agent
"""

from __future__ import annotations

import uuid

from ccsm_eval.trajectories.base import Token, Trajectory
from ccsm_eval.trajectories.concordia.scenarios import EpisodeLog


def episode_to_trajectory(
    log: EpisodeLog,
    quality_level: str,
    character_prompt: str = "",
) -> Trajectory:
    """Convert an EpisodeLog to a Trajectory in the standard CCSM format.

    Args:
        log: The episode log produced by a scenario's run() method.
        quality_level: The quality level string to embed in the trajectory.
        character_prompt: The evaluation character prompt (prepended at eval time).

    Returns:
        A Trajectory with Token objects and quality_scores populated.
    """
    tokens: list[Token] = []

    for position, entry in enumerate(log.entries):
        token = Token(
            text=entry.text,
            token_ids=[],
            is_observation=not entry.is_focal_action,
            semantic_type=entry.semantic_type,
            position=position,
        )
        tokens.append(token)

    n_obs = sum(1 for t in tokens if t.is_observation)
    n_act = sum(1 for t in tokens if not t.is_observation)

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        tokens=tokens,
        character_prompt=character_prompt,
        quality_scores=dict(log.quality_scores),
        quality_level=quality_level,
        environment="concordia",
        metadata={
            **log.metadata,
            "scenario_name": log.scenario_name,
            "focal_agent_name": log.focal_agent_name,
            "focal_agent_role": quality_level,
            "quality_level": quality_level,
            "n_obs_tokens": n_obs,
            "n_act_tokens": n_act,
            "n_turns": len(log.entries),
        },
    )


def validate_trajectory(traj: Trajectory) -> list[str]:
    """Sanity-check a trajectory, returning a list of warning strings (empty = OK)."""
    warnings: list[str] = []

    if not traj.tokens:
        warnings.append("Trajectory has no tokens.")
        return warnings

    n_act = sum(1 for t in traj.tokens if not t.is_observation)
    if n_act == 0:
        warnings.append("No action tokens found — focal agent never spoke.")

    n_obs = sum(1 for t in traj.tokens if t.is_observation)
    if n_obs == 0:
        warnings.append("No observation tokens found.")

    setup_count = sum(1 for t in traj.tokens if t.semantic_type == "setup")
    if setup_count == 0:
        warnings.append("No setup token found.")

    outcome_count = sum(1 for t in traj.tokens if t.semantic_type == "outcome")
    if outcome_count == 0:
        warnings.append("No outcome token found — game may not have terminated properly.")

    for metric in ("payoff", "social_welfare", "cooperation_score"):
        if metric not in traj.quality_scores:
            warnings.append(f"Missing quality metric: {metric!r}")

    return warnings
