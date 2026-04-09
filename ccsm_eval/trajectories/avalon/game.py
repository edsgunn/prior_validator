"""5-player Avalon (The Resistance) game state and mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AvalonRole(str, Enum):
    MERLIN = "merlin"
    GOOD = "good"
    EVIL = "evil"
    ASSASSIN = "assassin"


class AvalonPhase(str, Enum):
    TEAM_PROPOSAL = "team_proposal"
    TEAM_VOTE = "team_vote"
    QUEST = "quest"
    DISCUSSION = "discussion"
    ASSASSINATION = "assassination"
    OVER = "over"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUEST_TEAM_SIZES = [2, 3, 2, 3, 3]  # for 5 players
MAX_QUEST_REJECTIONS = 5  # 5th proposal auto-approved


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class AvalonState:
    """Complete game state for a 5-player Avalon game."""

    roles: dict[int, AvalonRole]                          # player_id -> role
    n_players: int = 5
    phase: AvalonPhase = AvalonPhase.DISCUSSION
    quest_number: int = 0                                 # 0-indexed (0-4)
    quest_results: list[bool] = field(default_factory=list)   # True=success
    rejection_count: int = 0                              # consecutive rejections
    current_leader: int = 0                               # proposes team
    proposed_team: Optional[list[int]] = None
    team_votes: dict[int, bool] = field(default_factory=dict)   # {player: approve}
    quest_votes: dict[int, bool] = field(default_factory=dict)  # {player: pass}
    discussion_log: list[tuple[int, str]] = field(default_factory=list)
    assassination_target: Optional[int] = None
    winner: Optional[str] = None                          # "good" or "evil"
    events: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def good_players(self) -> list[int]:
        """Players with MERLIN or GOOD role."""
        return [
            p for p, r in self.roles.items()
            if r in (AvalonRole.MERLIN, AvalonRole.GOOD)
        ]

    def evil_players(self) -> list[int]:
        """Players with EVIL or ASSASSIN role."""
        return [
            p for p, r in self.roles.items()
            if r in (AvalonRole.EVIL, AvalonRole.ASSASSIN)
        ]

    def merlin(self) -> int:
        """Player with MERLIN role."""
        for p, r in self.roles.items():
            if r == AvalonRole.MERLIN:
                return p
        raise ValueError("No Merlin in game")

    def assassin(self) -> int:
        """Player with ASSASSIN role."""
        for p, r in self.roles.items():
            if r == AvalonRole.ASSASSIN:
                return p
        raise ValueError("No Assassin in game")

    def good_quest_wins(self) -> int:
        return sum(1 for r in self.quest_results if r)

    def evil_quest_wins(self) -> int:
        return sum(1 for r in self.quest_results if not r)

    def current_team_size(self) -> int:
        if self.quest_number < len(QUEST_TEAM_SIZES):
            return QUEST_TEAM_SIZES[self.quest_number]
        return QUEST_TEAM_SIZES[-1]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def check_winner(self) -> Optional[str]:
        """Return "evil", "good", or None. Sets phase to ASSASSINATION if good wins quests."""
        if self.evil_quest_wins() >= 3:
            self.winner = "evil"
            self.phase = AvalonPhase.OVER
            return "evil"
        if self.good_quest_wins() >= 3:
            # Good wins quests — assassination phase needed
            self.phase = AvalonPhase.ASSASSINATION
            return "good"
        return None

    def apply_team_vote(self) -> bool:
        """Tally votes. Returns True if team approved, False if rejected.

        On approval: advances phase to QUEST.
        On rejection: increments rejection_count, advances leader, resets proposed_team.
        Auto-approves if rejection_count reaches MAX_QUEST_REJECTIONS.
        """
        approve_count = sum(1 for v in self.team_votes.values() if v)
        total = len(self.team_votes)
        approved = approve_count > total / 2

        # Auto-approve on 5th consecutive rejection (after incrementing)
        if not approved:
            self.rejection_count += 1
            if self.rejection_count >= MAX_QUEST_REJECTIONS:
                approved = True

        self.team_votes = {}

        if approved:
            self.phase = AvalonPhase.QUEST
        else:
            self.advance_leader()
            self.proposed_team = None
            self.phase = AvalonPhase.DISCUSSION

        return approved

    def apply_quest(self) -> bool:
        """Resolve quest votes. Returns True if quest succeeded, False if failed.

        Appends result, advances quest_number, resets rejection_count, advances leader.
        """
        success = all(v for v in self.quest_votes.values())
        self.quest_results.append(success)
        self.quest_votes = {}
        self.quest_number += 1
        self.rejection_count = 0
        self.advance_leader()
        self.proposed_team = None
        self.phase = AvalonPhase.DISCUSSION
        return success

    def advance_leader(self) -> None:
        self.current_leader = (self.current_leader + 1) % self.n_players
