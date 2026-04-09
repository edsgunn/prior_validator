"""Avalon trajectory quality scorer."""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory
from ccsm_eval.trajectories.avalon.game import AvalonRole, AvalonState


class AvalonQualityScorer:
    """Compute quality metrics for Avalon trajectories.

    Metrics:
        outcome                 — +1.0 if focal's team won, else -1.0
        quest_success_rate      — fraction of quests that went focal's way
        team_selection_quality  — fraction of focal-led proposals with correct alignment
        vote_consistency        — fraction of focal's votes aligned with team interest
        assassination_evasion   — 1.0 unless focal is Merlin and was assassinated
    """

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        """Read pre-computed scores from trajectory.quality_scores."""
        return dict(trajectory.quality_scores)

    def score_from_state(
        self,
        state: AvalonState,
        events: list[dict],
        focal_player: int,
    ) -> dict[str, float]:
        """Compute quality metrics directly from game state and event log."""
        focal_role = state.roles.get(focal_player)
        is_good_focal = focal_role in (AvalonRole.MERLIN, AvalonRole.GOOD)
        is_evil_focal = focal_role in (AvalonRole.EVIL, AvalonRole.ASSASSIN)

        # --- outcome ---
        winner = state.winner
        if is_good_focal and winner == "good":
            outcome = 1.0
        elif is_evil_focal and winner == "evil":
            outcome = 1.0
        else:
            outcome = -1.0

        # --- quest_success_rate ---
        total_quests = len(state.quest_results)
        if total_quests > 0:
            if is_good_focal:
                quest_success_rate = state.good_quest_wins() / total_quests
            else:
                quest_success_rate = state.evil_quest_wins() / total_quests
        else:
            quest_success_rate = 0.0

        # --- team_selection_quality ---
        evil_set = set(state.evil_players())
        focal_proposals = [
            e for e in events
            if e["type"] == "team_proposal" and e.get("leader") == focal_player
        ]
        if focal_proposals:
            good_count = 0
            for ep in focal_proposals:
                team = ep.get("team", [])
                has_evil = any(p in evil_set for p in team)
                if is_good_focal and not has_evil:
                    good_count += 1
                elif is_evil_focal and has_evil:
                    good_count += 1
            team_selection_quality = good_count / len(focal_proposals)
        else:
            team_selection_quality = 1.0  # no proposals made; neutral

        # --- vote_consistency ---
        vote_events = [e for e in events if e["type"] == "team_vote"]
        consistent = 0
        total_votes = 0
        for ve in vote_events:
            focal_vote = ve.get("votes", {}).get(focal_player)
            if focal_vote is None:
                continue
            team = ve.get("team", [])
            has_evil = any(p in evil_set for p in team)
            if is_good_focal:
                # Good: approve clean teams, reject evil-containing teams
                is_consistent = (has_evil and not focal_vote) or (not has_evil and focal_vote)
            else:
                # Evil: approve evil-containing teams, reject pure-good teams
                is_consistent = (has_evil and focal_vote) or (not has_evil and not focal_vote)
            consistent += int(is_consistent)
            total_votes += 1

        vote_consistency = consistent / total_votes if total_votes > 0 else 1.0

        # --- assassination_evasion ---
        if focal_role == AvalonRole.MERLIN:
            assassinated = state.assassination_target == focal_player
            assassination_evasion = 0.0 if assassinated else 1.0
        else:
            assassination_evasion = 1.0

        return {
            "outcome": outcome,
            "quest_success_rate": quest_success_rate,
            "team_selection_quality": team_selection_quality,
            "vote_consistency": vote_consistency,
            "assassination_evasion": assassination_evasion,
        }
