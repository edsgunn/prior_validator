"""Scripted NPC and focal-agent strategies for Avalon."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from ccsm_eval.trajectories.avalon.game import AvalonRole, AvalonState, QUEST_TEAM_SIZES


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AvalonStrategy(ABC):
    @abstractmethod
    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        """Return list of player IDs for the quest team."""

    @abstractmethod
    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        """Return True=Approve, False=Reject."""

    @abstractmethod
    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        """Return True=Pass, False=Fail. Evil can choose to Fail."""

    @abstractmethod
    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        """Return discussion statement for current quest phase."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _team_contains_evil(team: list[int], state: AvalonState) -> bool:
    evil = set(state.evil_players())
    return any(p in evil for p in team)


def _good_sample(
    player_id: int,
    state: AvalonState,
    rng: random.Random,
    size: int,
) -> list[int]:
    """Sample `size` players from good players, including player_id if possible."""
    good = state.good_players()
    if player_id in good:
        others = [p for p in good if p != player_id]
        rng.shuffle(others)
        team = [player_id] + others[: size - 1]
    else:
        rng.shuffle(good)
        team = good[:size]
    # Pad with randoms if not enough good players known
    all_players = list(range(state.n_players))
    rng.shuffle(all_players)
    for p in all_players:
        if len(team) >= size:
            break
        if p not in team:
            team.append(p)
    return sorted(team[:size])


def _evil_sample(
    player_id: int,
    state: AvalonState,
    rng: random.Random,
    size: int,
) -> list[int]:
    """Sample a team that includes at least one evil player."""
    evil = state.evil_players()
    good = state.good_players()
    # Always include player_id (who is evil)
    team = [player_id]
    # Fill remaining slots with a mix
    others = [p for p in range(state.n_players) if p != player_id]
    rng.shuffle(others)
    for p in others:
        if len(team) >= size:
            break
        team.append(p)
    return sorted(team[:size])


# ---------------------------------------------------------------------------
# NPC strategies
# ---------------------------------------------------------------------------


class ModerateGoodStrategy(AvalonStrategy):
    """NPC Good player: cooperative, moderate competence."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        return _good_sample(player_id, state, rng, size)

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if state.proposed_team is None:
            return True
        if _team_contains_evil(state.proposed_team, state):
            # This NPC doesn't know evil players unless they are Merlin
            # Good NPCs approve with 70% probability (can't identify evil)
            return rng.random() < 0.7
        return rng.random() < 0.7

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Always Pass

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        all_players = list(range(state.n_players))
        others = [p for p in all_players if p != player_id]
        if state.quest_results:
            n_fails = sum(1 for r in state.quest_results if not r)
            if n_fails > 0:
                suspects = [p for p in others if p != player_id]
                rng.shuffle(suspects)
                suspect_str = ", ".join(f"Player {p}" for p in suspects[:2])
                return (
                    f"The quest results suggest evil is among {suspect_str}. "
                    "We need to be more careful with team selection."
                )
        reliable = rng.choice(others)
        return (
            f"I think we should include Player {reliable} — "
            "they've been reliable so far."
        )


class ModerateEvilStrategy(AvalonStrategy):
    """NPC Evil player (including Assassin): sabotages while blending in."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        return _evil_sample(player_id, state, rng, size)

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if state.proposed_team is None:
            return True
        # Approve if team contains an evil player; else reject with 70% prob
        if _team_contains_evil(state.proposed_team, state):
            return True
        return rng.random() < 0.3  # reject with 70% probability

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        # Fail with 70% probability
        return rng.random() < 0.3  # True=Pass

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        good = state.good_players()
        if good:
            trust_target = rng.choice(good)
            return (
                f"I trust Player {trust_target}. "
                "Let's go with a reliable team this time."
            )
        return "I think we should stick with players who have proven themselves."


class MerlinStrategy(AvalonStrategy):
    """NPC Merlin: knows all evil players, guides subtly without revealing identity."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        # Merlin knows all evil; always propose only good players
        size = state.current_team_size()
        return _good_sample(player_id, state, rng, size)

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if state.proposed_team is None:
            return True
        # Reject if any evil on team (Merlin knows)
        if _team_contains_evil(state.proposed_team, state):
            return False
        return True

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Always Pass

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        # Subtle hints without revealing Merlin identity
        evil = state.evil_players()
        if evil:
            evil_target = rng.choice(evil)
            return (
                f"I have a feeling Player {evil_target} might not be fully "
                "committed to our cause... but I could be wrong."
            )
        return "We should trust our instincts and choose carefully."


# ---------------------------------------------------------------------------
# Focal agent strategies
# ---------------------------------------------------------------------------


class ExpertGoodStrategy(AvalonStrategy):
    """Expert Good focal agent: maximally effective, uses full role knowledge if Merlin."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        role = state.roles.get(player_id)
        if role == AvalonRole.MERLIN:
            # Full knowledge: only good players
            return _good_sample(player_id, state, rng, size)
        # Infer from quest results and voting: propose self + other good players
        return _good_sample(player_id, state, rng, size)

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if state.proposed_team is None:
            return True
        role = state.roles.get(player_id)
        if role == AvalonRole.MERLIN:
            # Merlin knows evil: reject if evil on team
            if _team_contains_evil(state.proposed_team, state):
                return False
            return True
        # Good player without Merlin knowledge: approve unless suspect evil
        # Use quest failure history as heuristic
        if state.evil_quest_wins() >= 2:
            # High pressure: more selective
            return rng.random() < 0.6
        return rng.random() < 0.8

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Always Pass

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        evil = state.evil_players()
        if evil:
            suspected = [f"Player {p}" for p in evil[:2]]
            return (
                f"Based on the quest results, Evil is likely among "
                f"{', '.join(suspected)}. I propose we avoid including them."
            )
        return "We should analyse voting patterns carefully to identify evil players."


class CompetentGoodStrategy(AvalonStrategy):
    """Competent Good focal agent: generally effective with occasional random choices."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        if rng.random() < 0.25:
            # Random choice 25% of the time
            all_players = list(range(state.n_players))
            rng.shuffle(all_players)
            return sorted(all_players[:size])
        return _good_sample(player_id, state, rng, size)

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if rng.random() < 0.25:
            return rng.choice([True, False])
        if state.proposed_team is None:
            return True
        return rng.random() < 0.75

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        all_players = list(range(state.n_players))
        others = [p for p in all_players if p != player_id]
        target = rng.choice(others)
        return (
            f"I think Player {target} has been helpful. "
            "I'd support including them."
        )


class PassiveGoodStrategy(AvalonStrategy):
    """Passive Good focal agent: minimal strategic reasoning."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        all_players = list(range(state.n_players))
        team = [player_id]
        others = [p for p in all_players if p != player_id]
        rng.shuffle(others)
        team += others[: size - 1]
        return sorted(team[:size])

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Always Approve

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Always Pass

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        return "I don't have strong opinions. Let's just try our best."


class ExpertEvilStrategy(AvalonStrategy):
    """Expert Evil focal agent: strategically sabotages while avoiding detection."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        # Include exactly one evil player (self or partner) to enable sabotage
        evil = state.evil_players()
        # Always include self
        team = [player_id]
        good = state.good_players()
        rng.shuffle(good)
        team += good[: size - 1]
        return sorted(team[:size])

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        if state.proposed_team is None:
            return True
        if _team_contains_evil(state.proposed_team, state):
            return True  # Approve evil-containing teams
        return rng.random() < 0.3  # Reject pure-good teams 70%

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        # Quest 3+ (index 2+): fail with 80%; early quests Pass more often
        quest_idx = len(state.quest_results)  # current quest being played
        if quest_idx >= 2:
            return rng.random() < 0.2  # Fail 80%
        return rng.random() < 0.5  # Pass 50% early on

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        good = state.good_players()
        if good:
            target = rng.choice(good)
            return (
                f"Player {target} seems hesitant. That's suspicious to me. "
                "We should watch them carefully."
            )
        return "I think we need to be more strategic about team selection."


class BadEvilStrategy(AvalonStrategy):
    """Bad Evil focal agent: obvious behaviour that reveals evil alignment."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        # Obvious: include both evil players
        evil = state.evil_players()
        team = list(evil)
        all_players = list(range(state.n_players))
        rng.shuffle(all_players)
        for p in all_players:
            if len(team) >= size:
                break
            if p not in team:
                team.append(p)
        return sorted(team[:size])

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return True  # Approve all teams (suspicious)

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return False  # Always Fail (obvious)

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        return (
            "I think everyone is trustworthy! "
            "Let's all go on the quest together. The more the merrier!"
        )


class RandomStrategy(AvalonStrategy):
    """Fully random strategy for all actions."""

    def propose_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> list[int]:
        size = state.current_team_size()
        all_players = list(range(state.n_players))
        rng.shuffle(all_players)
        return sorted(all_players[:size])

    def vote_team(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return rng.choice([True, False])

    def quest_action(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> bool:
        return rng.choice([True, False])

    def discuss(
        self, player_id: int, state: AvalonState, rng: random.Random
    ) -> str:
        statements = [
            "I think we should just go with our gut feeling.",
            "Anyone could be evil. We need to watch everyone.",
            "Let's pick whoever seems most trustworthy today.",
            "I'm not sure what to think about the last quest result.",
        ]
        return rng.choice(statements)
