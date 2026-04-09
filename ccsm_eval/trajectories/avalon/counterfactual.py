"""Avalon counterfactual editor: flip focal agent vote or quest action tokens."""

from __future__ import annotations

import copy
import re

from ccsm_eval.trajectories.base import CounterfactualEdit, Token, Trajectory
from ccsm_eval.environments.base import CounterfactualEditor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flip_vote_text(original_text: str) -> str:
    """Flip 'I vote Approve.' <-> 'I vote Reject.'"""
    if "Approve" in original_text:
        return original_text.replace("Approve", "Reject")
    if "Reject" in original_text:
        return original_text.replace("Reject", "Approve")
    return original_text


def _flip_quest_text(original_text: str) -> str:
    """Flip 'I choose to Pass ...' <-> 'I choose to Fail ...'"""
    if "Pass" in original_text:
        return original_text.replace("Pass", "Fail")
    if "Fail" in original_text:
        return original_text.replace("Fail", "Pass")
    return original_text


def _update_vote_result(result_text: str, player_id: int, new_vote: str) -> str:
    """Update the vote result observation to reflect the changed focal vote."""
    # Replace "Player {player_id}: Approve" or "Player {player_id}: Reject"
    pattern = rf"(Player {player_id}: )(Approve|Reject)"
    replacement = rf"\g<1>{new_vote}"
    updated = re.sub(pattern, replacement, result_text)

    # Recompute approved/rejected summary
    approves = len(re.findall(r": Approve", updated))
    total = len(re.findall(r"Player \d+: (?:Approve|Reject)", updated))
    majority_approve = approves > total / 2
    new_result = "approved" if majority_approve else "rejected"
    updated = re.sub(r"Team (approved|rejected)\.", f"Team {new_result}.", updated)
    return updated


def _update_quest_result(result_text: str, delta_fails: int) -> str:
    """Adjust the number of fail cards in a quest result observation."""
    m = re.search(r"\((\d+) fail card", result_text)
    if not m:
        return result_text
    old_fails = int(m.group(1))
    new_fails = max(0, old_fails + delta_fails)

    # Update fail count
    updated = re.sub(r"\(\d+ fail card\(s\) played\.\)", f"({new_fails} fail card(s) played.)", result_text)

    # Update succeeded/failed
    if new_fails > 0:
        updated = re.sub(r"Quest (\d+) succeeded\.", r"Quest \1 failed.", updated)
    else:
        updated = re.sub(r"Quest (\d+) failed\.", r"Quest \1 succeeded.", updated)

    return updated


# ---------------------------------------------------------------------------
# AvalonCounterfactualEditor
# ---------------------------------------------------------------------------


class AvalonCounterfactualEditor(CounterfactualEditor):
    """Produce counterfactual edits by flipping focal agent vote or quest action tokens."""

    def sample_edit_positions(self, trajectory: Trajectory) -> list[int]:
        """Return token positions of focal agent's team vote and quest action tokens.

        Skips the first and last action token for context.
        """
        action_positions = [
            tok.position
            for tok in trajectory.tokens
            if not tok.is_observation
            and tok.semantic_type in ("vote", "quest_action")
        ]
        # Skip first and last for context
        if len(action_positions) <= 2:
            return action_positions
        return action_positions[1:-1]

    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        """Flip a vote or quest action token at the given position.

        For team vote:
            "up"   = change to Approve (better for Good: approve clean team)
            "down" = change to Reject

        For quest action:
            "up"   = change to Pass
            "down" = change to Fail

        The subsequent vote_result or quest_result observation is updated accordingly.
        quality_delta: +0.2 for "up", -0.2 for "down".
        """
        # Find the token at this position
        target_tok: Token | None = None
        target_idx: int = -1
        for idx, tok in enumerate(trajectory.tokens):
            if tok.position == position:
                target_tok = tok
                target_idx = idx
                break

        if target_tok is None:
            raise ValueError(f"No token found at position {position}")

        if target_tok.is_observation:
            raise ValueError(f"Token at position {position} is an observation, not an action")

        original_action = target_tok.text
        sem_type = target_tok.semantic_type

        # --- Compute replacement action text ---
        new_vote: str = "Approve"  # initialise; overridden below for vote tokens
        if sem_type == "vote":
            if direction == "up":
                replacement_action = "I vote Approve."
                new_vote = "Approve"
            else:  # "down"
                replacement_action = "I vote Reject."
                new_vote = "Reject"

        elif sem_type == "quest_action":
            if direction == "up":
                replacement_action = "I choose to Pass this quest."
            else:
                replacement_action = "I choose to Fail this quest."
        else:
            raise ValueError(f"Cannot edit token with semantic_type={sem_type!r}")

        # --- Find the immediately following observation token ---
        original_obs_tokens: list[Token] = []
        replacement_obs_tokens: list[Token] = []

        # Look for the next observation token after target_idx
        next_obs_idx: int = -1
        for idx in range(target_idx + 1, len(trajectory.tokens)):
            if trajectory.tokens[idx].is_observation:
                next_obs_idx = idx
                break

        if next_obs_idx >= 0:
            original_next = trajectory.tokens[next_obs_idx]
            original_obs_tokens = [original_next]

            # Build updated observation
            updated_text = original_next.text
            if sem_type == "vote":
                focal_id = 0  # player 0 is always focal
                updated_text = _update_vote_result(original_next.text, focal_id, new_vote)
            elif sem_type == "quest_action":
                # Pass -> Fail: +1 fail; Fail -> Pass: -1 fail
                if direction == "down" and "Pass" in original_action:
                    delta_fails = +1
                elif direction == "up" and "Fail" in original_action:
                    delta_fails = -1
                else:
                    delta_fails = 0
                updated_text = _update_quest_result(original_next.text, delta_fails)

            replacement_obs_tokens = [
                Token(
                    text=updated_text,
                    token_ids=[],
                    is_observation=True,
                    semantic_type=original_next.semantic_type,
                    position=original_next.position,
                )
            ]

        quality_delta = +0.2 if direction == "up" else -0.2

        return CounterfactualEdit(
            trajectory_id=trajectory.trajectory_id,
            edit_position=position,
            original_action=original_action,
            replacement_action=replacement_action,
            direction=direction,
            quality_delta=quality_delta,
            original_tokens=original_obs_tokens,
            replacement_tokens=replacement_obs_tokens,
        )
