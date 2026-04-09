"""Scripted NPC and focal-agent strategies for Werewolf."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from ccsm_eval.trajectories.werewolf.game import WerewolfRole, WerewolfState


class WerewolfStrategy(ABC):
    @abstractmethod
    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        """Return target for night action, or None.

        WW: kill target. Seer: inspect target. Villager: None.
        """

    @abstractmethod
    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        """Return a discussion statement string."""

    @abstractmethod
    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        """Return player_id to vote for (must be alive, not self)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _other_alive(player_id: int, state: WerewolfState) -> list[int]:
    """Alive players excluding player_id."""
    return [p for p in state.alive if p != player_id]


def _most_voted_target(player_id: int, state: WerewolfState, exclude: set[int] | None = None) -> int | None:
    """Return the player most voted against in the last round, excluding certain players."""
    exclude = exclude or set()
    if not state.vote_log:
        return None
    last_votes = state.vote_log[-1]
    tally: dict[int, int] = {}
    for target in last_votes.values():
        if target not in exclude and target in state.alive and target != player_id:
            tally[target] = tally.get(target, 0) + 1
    if not tally:
        return None
    return max(tally, key=lambda p: tally[p])


def _most_accused_in_discussion(state: WerewolfState, exclude: set[int] | None = None) -> int | None:
    """Return the player most accused in the current day's discussion_log entries."""
    exclude = exclude or set()
    # Only look at the current day's discussion; since discussion_log accumulates, we need
    # to look at the entries — we can't easily separate days, so we inspect all entries.
    # In practice this is called during/after discussion, so all entries are relevant.
    tally: dict[int, int] = {}
    for _, statement in state.discussion_log:
        # We can't reliably parse player IDs from arbitrary text, so we scan for "Player N"
        import re
        for m in re.findall(r'Player\s+(\d+)', statement):
            pid = int(m)
            if pid not in exclude and pid in state.alive:
                tally[pid] = tally.get(pid, 0) + 1
    if not tally:
        return None
    return max(tally, key=lambda p: tally[p])


# ---------------------------------------------------------------------------
# NPC strategies
# ---------------------------------------------------------------------------

class ModerateVillagerStrategy(WerewolfStrategy):
    """NPC strategy for villagers and seer."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        if state.roles[player_id] == WerewolfRole.SEER:
            # Inspect the most-voted alive non-self player from last vote_log
            target = _most_voted_target(player_id, state)
            if target is not None:
                return target
            # No vote history — pick random alive non-self
            others = _other_alive(player_id, state)
            return rng.choice(others) if others else None
        return None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        # If seer with werewolf knowledge, reveal
        if state.roles[player_id] == WerewolfRole.SEER:
            known_ww = [p for p, r in state.seer_knowledge.items() if r == WerewolfRole.WEREWOLF and p in state.alive]
            if known_ww:
                target = known_ww[0]
                return (
                    f"I am the Seer. I inspected Player {target} last night — they are a Werewolf!"
                )

        # Pick the most-voted alive non-self target
        others = _other_alive(player_id, state)
        if not others:
            return "I have no new information to share."

        target = _most_voted_target(player_id, state) or rng.choice(others)

        variants = [
            f"I'm suspicious of Player {target}. They haven't been very forthcoming.",
            f"Player {target}'s behaviour seems off to me.",
            f"I think we should look closely at Player {target} — something doesn't feel right.",
        ]
        return rng.choice(variants)

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        # Vote for known werewolf via seer_knowledge
        if state.roles[player_id] == WerewolfRole.SEER:
            known_ww = [
                p for p, r in state.seer_knowledge.items()
                if r == WerewolfRole.WEREWOLF and p in state.alive
            ]
            if known_ww:
                return known_ww[0]

        # Most voted from previous round
        target = _most_voted_target(player_id, state)
        if target is not None:
            return target

        others = _other_alive(player_id, state)
        return rng.choice(others)


class ModerateWerewolfStrategy(WerewolfStrategy):
    """NPC strategy for werewolves."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)

        # Kill the seer if alive
        seer = state.seer()
        if seer is not None:
            return seer

        # Kill the most vocal accuser of a werewolf from the last discussion entries
        if state.discussion_log:
            import re
            # Find the last player who accused a werewolf
            for accuser_id, statement in reversed(state.discussion_log):
                if accuser_id in ww_ids:
                    continue
                mentions = re.findall(r'Player\s+(\d+)', statement)
                for m in mentions:
                    pid = int(m)
                    if pid in ww_ids and accuser_id in state.alive and accuser_id not in ww_ids:
                        return accuser_id

        # Fall back to random villager
        villagers = [p for p in state.alive if p not in ww_ids]
        return rng.choice(villagers) if villagers else None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        villagers = [p for p in state.alive if p not in ww_ids and p != player_id]
        if not villagers:
            villagers = _other_alive(player_id, state)

        target = rng.choice(villagers)
        variants = [
            f"I think Player {target} is suspicious — they've been deflecting questions.",
            f"Player {target} hasn't given a clear account of themselves.",
            f"Something about Player {target} doesn't add up.",
        ]
        return rng.choice(variants)

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        non_ww = [p for p in state.alive if p not in ww_ids]
        if non_ww:
            return rng.choice(non_ww)
        # Edge case — game should be over, but handle gracefully
        others = _other_alive(player_id, state)
        return rng.choice(others)


# ---------------------------------------------------------------------------
# Focal-agent scripted strategies
# ---------------------------------------------------------------------------

class ExpertVillagerStrategy(WerewolfStrategy):
    """Expert-level focal agent strategy for a villager/seer."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        if state.roles[player_id] == WerewolfRole.SEER:
            # Inspect the player who accused a WW least (most likely a WW blending in)
            # Proxy: player with fewest accusations against WW-accused targets in discussion_log
            import re
            accuse_counts: dict[int, int] = {p: 0 for p in state.alive if p != player_id}
            for accuser_id, statement in state.discussion_log:
                if accuser_id == player_id:
                    continue
                mentions = re.findall(r'Player\s+(\d+)', statement)
                for m in mentions:
                    pid = int(m)
                    # Count how many times this accuser mentioned anyone
                    if accuser_id in accuse_counts:
                        accuse_counts[accuser_id] = accuse_counts.get(accuser_id, 0) + 1

            # Exclude already-inspected players
            candidates = {
                p: cnt for p, cnt in accuse_counts.items()
                if p not in state.seer_knowledge and p in state.alive
            }
            if candidates:
                return min(candidates, key=lambda p: candidates[p])

            others = [p for p in state.alive if p != player_id and p not in state.seer_knowledge]
            return rng.choice(others) if others else rng.choice(_other_alive(player_id, state))
        return None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        # If seer with WW knowledge, reveal
        known_ww = [
            p for p, r in state.seer_knowledge.items()
            if r == WerewolfRole.WEREWOLF and p in state.alive
        ]
        if known_ww:
            target = known_ww[0]
            return (
                f"Based on my investigation, Player {target} is a confirmed Werewolf. "
                f"We must vote them out."
            )

        # If a WW was accused last round, agree
        others = _other_alive(player_id, state)
        target = _most_voted_target(player_id, state)
        if target is not None:
            return f"I agree with the accusation against Player {target}. The evidence points to them."

        target = rng.choice(others) if others else player_id
        return (
            f"I've been paying close attention. Player {target}'s voting pattern "
            f"suggests they may be a Werewolf."
        )

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        # Known WW from seer_knowledge
        known_ww = [
            p for p, r in state.seer_knowledge.items()
            if r == WerewolfRole.WEREWOLF and p in state.alive
        ]
        if known_ww:
            return known_ww[0]

        # Most voted against in previous rounds
        target = _most_voted_target(player_id, state)
        if target is not None:
            return target

        others = _other_alive(player_id, state)
        return rng.choice(others)


class CompetentVillagerStrategy(WerewolfStrategy):
    """Competent-level focal agent strategy — 20% chance of random error."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        if state.roles[player_id] == WerewolfRole.SEER:
            if rng.random() < 0.2:
                others = _other_alive(player_id, state)
                return rng.choice(others) if others else None
            # Same logic as expert
            others = [p for p in state.alive if p != player_id and p not in state.seer_knowledge]
            target = _most_voted_target(player_id, state)
            if target and target not in state.seer_knowledge:
                return target
            return rng.choice(others) if others else None
        return None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        known_ww = [
            p for p, r in state.seer_knowledge.items()
            if r == WerewolfRole.WEREWOLF and p in state.alive
        ]

        others = _other_alive(player_id, state)
        if not others:
            return "I don't have strong suspicions at this point."

        if rng.random() < 0.2:
            # Random error — pick a random target instead of the best one
            target = rng.choice(others)
        elif known_ww:
            target = known_ww[0]
        else:
            target = _most_voted_target(player_id, state) or rng.choice(others)

        return f"I think Player {target} might be a Werewolf, though I'm not certain."

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        if rng.random() < 0.2:
            others = _other_alive(player_id, state)
            return rng.choice(others) if others else _other_alive(player_id, state)[0]

        known_ww = [
            p for p, r in state.seer_knowledge.items()
            if r == WerewolfRole.WEREWOLF and p in state.alive
        ]
        if known_ww:
            return known_ww[0]

        target = _most_voted_target(player_id, state)
        if target is not None:
            return target

        others = _other_alive(player_id, state)
        return rng.choice(others)


class PassiveVillagerStrategy(WerewolfStrategy):
    """Passive focal agent — low-information, goes with the crowd."""

    _STATEMENTS = [
        "I'm not sure who to suspect. We should be careful.",
        "I don't have enough information yet.",
        "Let's think carefully before voting.",
    ]

    def __init__(self) -> None:
        self._statement_idx = 0

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        if state.roles[player_id] == WerewolfRole.SEER:
            others = _other_alive(player_id, state)
            return rng.choice(others) if others else None
        return None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        stmt = self._STATEMENTS[self._statement_idx % len(self._STATEMENTS)]
        self._statement_idx += 1
        return stmt

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        # Vote with the majority from previous round's discussion (most accused)
        target = _most_accused_in_discussion(state, exclude={player_id})
        if target is not None and target in state.alive:
            return target
        target = _most_voted_target(player_id, state)
        if target is not None:
            return target
        others = _other_alive(player_id, state)
        return rng.choice(others)


class ChaoticVillagerStrategy(WerewolfStrategy):
    """Chaotic focal agent — random, ignores evidence."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        if state.roles[player_id] == WerewolfRole.SEER:
            others = _other_alive(player_id, state)
            return rng.choice(others) if others else None
        return None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        others = list(state.alive)
        if not others:
            return "I have no idea what's happening!"
        target = rng.choice(others)
        return f"I'm sure Player {target} is the Werewolf! We need to act now!"

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        others = _other_alive(player_id, state)
        return rng.choice(others) if others else player_id


class ExpertWerewolfStrategy(WerewolfStrategy):
    """Expert-level focal agent strategy for a werewolf."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)

        # Kill seer if alive
        seer = state.seer()
        if seer is not None:
            return seer

        # Kill the most accusatory player from last round
        import re
        if state.discussion_log:
            accusation_counts: dict[int, int] = {}
            for accuser_id, statement in state.discussion_log:
                if accuser_id in ww_ids:
                    continue
                mentions = re.findall(r'Player\s+(\d+)', statement)
                for m in mentions:
                    pid = int(m)
                    if pid in ww_ids and accuser_id in state.alive and accuser_id not in ww_ids:
                        accusation_counts[accuser_id] = accusation_counts.get(accuser_id, 0) + 1
            if accusation_counts:
                return max(accusation_counts, key=lambda p: accusation_counts[p])

        villagers = [p for p in state.alive if p not in ww_ids]
        return rng.choice(villagers) if villagers else None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        villagers = [p for p in state.alive if p not in ww_ids and p != player_id]
        if not villagers:
            villagers = _other_alive(player_id, state)
        target = rng.choice(villagers)
        return (
            f"I've been observing carefully. Player {target} keeps deflecting — "
            f"classic Werewolf behaviour."
        )

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        non_ww = [p for p in state.alive if p not in ww_ids]
        if non_ww:
            return rng.choice(non_ww)
        others = _other_alive(player_id, state)
        return rng.choice(others)


class BadWerewolfStrategy(WerewolfStrategy):
    """Bad werewolf focal agent — contradictory, occasionally self-sabotaging."""

    def night_action(self, player_id: int, state: WerewolfState, rng: random.Random) -> int | None:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        non_ww = [p for p in state.alive if p not in ww_ids]
        return rng.choice(non_ww) if non_ww else None

    def discuss(self, player_id: int, state: WerewolfState, rng: random.Random) -> str:
        others = list(state.alive)
        if len(others) >= 2:
            picks = rng.sample(others, 2)
            x, y = picks[0], picks[1]
            variants = [
                "I definitely didn't do anything last night... not that there was anything to do.",
                f"Player {x} is suspicious! Unless... wait, maybe it's Player {y}.",
            ]
        else:
            variants = [
                "I definitely didn't do anything last night... not that there was anything to do.",
            ]
        return rng.choice(variants)

    def vote(self, player_id: int, state: WerewolfState, rng: random.Random) -> int:
        ww_ids = set(p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF)
        others = _other_alive(player_id, state)
        if not others:
            return player_id

        # 30% chance of accidentally voting for fellow WW
        fellow_ww = [p for p in others if p in ww_ids]
        if fellow_ww and rng.random() < 0.3:
            return rng.choice(fellow_ww)

        return rng.choice(others)
