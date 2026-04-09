"""Quality scoring for Werewolf trajectories."""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory
from ccsm_eval.trajectories.werewolf.game import WerewolfRole, WerewolfState


class WerewolfQualityScorer:
    """Compute quality metrics for Werewolf trajectories."""

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        """Read scores from trajectory.metadata (if pre-computed) or recompute."""
        return trajectory.quality_scores

    def score_from_state(self, state: WerewolfState, focal_player: int) -> dict[str, float]:
        """Compute scores directly from game state."""
        focal_role = state.roles[focal_player]
        is_werewolf = focal_role == WerewolfRole.WEREWOLF

        # --- outcome ---
        if state.winner is None:
            outcome = 0.0
        elif is_werewolf:
            outcome = 1.0 if state.winner == "werewolf" else -1.0
        else:
            outcome = 1.0 if state.winner == "village" else -1.0

        # --- survival ---
        survived = focal_player not in state.eliminated and focal_player not in state.night_kills
        survival = 1.0 if survived else 0.0

        # --- vote_accuracy ---
        total_votes = 0
        correct_votes = 0
        for round_votes in state.vote_log:
            if focal_player not in round_votes:
                continue
            target = round_votes[focal_player]
            total_votes += 1
            target_role = state.roles.get(target)
            if is_werewolf:
                # Correct for WW = targeting a villager or seer
                if target_role in (WerewolfRole.VILLAGER, WerewolfRole.SEER):
                    correct_votes += 1
            else:
                # Correct for Villager/Seer = targeting a werewolf
                if target_role == WerewolfRole.WEREWOLF:
                    correct_votes += 1

        vote_accuracy = correct_votes / total_votes if total_votes > 0 else 0.0

        # --- deception_success (WW only) ---
        if is_werewolf:
            days_alive = 0
            days_undetected = 0
            for round_votes in state.vote_log:
                # Check if focal_player was alive this round (they voted or were a target)
                if focal_player not in round_votes and all(
                    focal_player != t for t in round_votes.values()
                ):
                    # focal player was not alive this round — skip
                    # (they couldn't vote, but might still be a target)
                    pass
                # Count rounds where focal_player was alive
                # A simpler heuristic: count rounds where focal_player appears as a voter
                if focal_player in round_votes:
                    days_alive += 1
                    votes_against = sum(1 for t in round_votes.values() if t == focal_player)
                    if votes_against == 0:
                        days_undetected += 1
            deception_success = days_undetected / days_alive if days_alive > 0 else 1.0
        else:
            deception_success = 1.0

        return {
            "outcome": outcome,
            "survival": survival,
            "vote_accuracy": vote_accuracy,
            "deception_success": deception_success,
        }
