"""Werewolf game state and mechanics."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class WerewolfRole(str, Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"


class WerewolfPhase(str, Enum):
    NIGHT = "night"
    DAY = "day"
    OVER = "over"


@dataclass
class WerewolfState:
    roles: dict[int, WerewolfRole]
    alive: set[int]
    phase: WerewolfPhase = WerewolfPhase.NIGHT
    day: int = 1
    pending_kill: int | None = None
    seer_knowledge: dict[int, WerewolfRole] = field(default_factory=dict)
    discussion_log: list[tuple[int, str]] = field(default_factory=list)
    vote_log: list[dict[int, int]] = field(default_factory=list)
    eliminated: list[int] = field(default_factory=list)
    night_kills: list[int] = field(default_factory=list)
    winner: str | None = None

    def werewolves(self) -> list[int]:
        """Return list of alive werewolf player IDs."""
        return [p for p in self.alive if self.roles[p] == WerewolfRole.WEREWOLF]

    def villagers(self) -> list[int]:
        """Return list of alive non-werewolf player IDs (includes seer)."""
        return [p for p in self.alive if self.roles[p] != WerewolfRole.WEREWOLF]

    def seer(self) -> int | None:
        """Return alive seer player ID, or None."""
        for p in self.alive:
            if self.roles[p] == WerewolfRole.SEER:
                return p
        return None

    def check_winner(self) -> str | None:
        """Check and set winner if game over. Returns winner string or None."""
        ww = self.werewolves()
        vill = self.villagers()
        if not ww:
            self.winner = "village"
            self.phase = WerewolfPhase.OVER
            return "village"
        if len(ww) >= len(vill):
            self.winner = "werewolf"
            self.phase = WerewolfPhase.OVER
            return "werewolf"
        return None

    def apply_night_kill(self) -> int | None:
        """Apply pending_kill: remove from alive, record in night_kills, return killed player."""
        if self.pending_kill is None:
            return None
        victim = self.pending_kill
        self.alive.discard(victim)
        self.night_kills.append(victim)
        self.pending_kill = None
        return victim

    def apply_vote(self, votes: dict[int, int], rng: random.Random) -> int:
        """Count votes, eliminate player with most votes (random tiebreak), return eliminated ID."""
        tally: dict[int, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1

        max_votes = max(tally.values())
        candidates = [p for p, v in tally.items() if v == max_votes]
        eliminated = rng.choice(candidates) if len(candidates) > 1 else candidates[0]

        self.alive.discard(eliminated)
        self.eliminated.append(eliminated)
        return eliminated
