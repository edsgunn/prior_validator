"""Phase 1: Generate Avalon trajectories at controlled quality levels."""

from __future__ import annotations

import random
import uuid
from typing import Optional

from ccsm_eval.trajectories.base import Token, Trajectory
from ccsm_eval.trajectories.avalon.game import (
    AvalonPhase,
    AvalonRole,
    AvalonState,
    QUEST_TEAM_SIZES,
    MAX_QUEST_REJECTIONS,
)
from ccsm_eval.trajectories.avalon.strategies import (
    AvalonStrategy,
    BadEvilStrategy,
    CompetentGoodStrategy,
    ExpertEvilStrategy,
    ExpertGoodStrategy,
    MerlinStrategy,
    ModerateEvilStrategy,
    ModerateGoodStrategy,
    PassiveGoodStrategy,
    RandomStrategy,
)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

_FOCAL_ROLE_MAP: dict[str, AvalonRole] = {
    "expert_good": AvalonRole.MERLIN,
    "competent_good": AvalonRole.GOOD,
    "passive_good": AvalonRole.GOOD,
    "expert_evil": AvalonRole.EVIL,
    "bad_evil": AvalonRole.EVIL,
    "random": AvalonRole.GOOD,
}

_FOCAL_STRATEGY_MAP: dict[str, type[AvalonStrategy]] = {
    "expert_good": ExpertGoodStrategy,
    "competent_good": CompetentGoodStrategy,
    "passive_good": PassiveGoodStrategy,
    "expert_evil": ExpertEvilStrategy,
    "bad_evil": BadEvilStrategy,
    "random": RandomStrategy,
}

QUALITY_LEVELS = [
    "expert_good",
    "competent_good",
    "passive_good",
    "expert_evil",
    "bad_evil",
    "random",
]


def _assign_roles(quality_level: str, rng: random.Random) -> dict[int, AvalonRole]:
    """Assign roles to 5 players. Player 0 is focal."""
    focal_role = _FOCAL_ROLE_MAP[quality_level]
    roles: dict[int, AvalonRole] = {0: focal_role}

    remaining_players = list(range(1, 5))
    rng.shuffle(remaining_players)

    # Determine what roles still need assigning
    # 5-player Avalon: 3 good (Merlin, Good, Good) + 2 evil (Assassin, Evil)
    needed_roles: list[AvalonRole] = []

    if focal_role == AvalonRole.MERLIN:
        # Focal is Merlin; need Assassin + Evil + 2 Good
        needed_roles = [AvalonRole.ASSASSIN, AvalonRole.EVIL, AvalonRole.GOOD, AvalonRole.GOOD]
    elif focal_role == AvalonRole.GOOD:
        # Focal is Good; need Merlin + Assassin + Evil + Good
        needed_roles = [AvalonRole.MERLIN, AvalonRole.ASSASSIN, AvalonRole.EVIL, AvalonRole.GOOD]
    elif focal_role == AvalonRole.EVIL:
        # Focal is Evil; need Merlin + Assassin + Good + Good
        needed_roles = [AvalonRole.MERLIN, AvalonRole.ASSASSIN, AvalonRole.GOOD, AvalonRole.GOOD]
    elif focal_role == AvalonRole.ASSASSIN:
        # Focal is Assassin; need Merlin + Evil + Good + Good
        needed_roles = [AvalonRole.MERLIN, AvalonRole.EVIL, AvalonRole.GOOD, AvalonRole.GOOD]

    rng.shuffle(needed_roles)
    for player, role in zip(remaining_players, needed_roles):
        roles[player] = role

    return roles


def _make_npc_strategy(player_id: int, state: AvalonState) -> AvalonStrategy:
    """Create appropriate NPC strategy based on role."""
    role = state.roles[player_id]
    if role == AvalonRole.MERLIN:
        return MerlinStrategy()
    elif role == AvalonRole.GOOD:
        return ModerateGoodStrategy()
    elif role in (AvalonRole.EVIL, AvalonRole.ASSASSIN):
        return ModerateEvilStrategy()
    return ModerateGoodStrategy()


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def _run_game(
    state: AvalonState,
    strategies: dict[int, AvalonStrategy],
    rng: random.Random,
    max_quests: int = 5,
) -> list[dict]:
    """Run a complete Avalon game and return the event log."""
    events: list[dict] = []

    while state.quest_number < max_quests and state.winner is None:
        # --- Discussion phase ---
        statements: list[tuple[int, str]] = []
        for p in range(state.n_players):
            stmt = strategies[p].discuss(p, state, rng)
            state.discussion_log.append((p, stmt))
            statements.append((p, stmt))
        events.append({
            "type": "discussion",
            "quest": state.quest_number + 1,
            "statements": statements,
        })

        # --- Team proposal (may loop on rejections) ---
        while True:
            leader = state.current_leader
            team = strategies[leader].propose_team(leader, state, rng)
            # Clamp to valid size
            size = state.current_team_size()
            team = list(dict.fromkeys(team))[:size]  # deduplicate, trim
            while len(team) < size:
                # pad if needed
                for p in range(state.n_players):
                    if p not in team:
                        team.append(p)
                    if len(team) >= size:
                        break
            state.proposed_team = team
            events.append({
                "type": "team_proposal",
                "quest": state.quest_number + 1,
                "leader": leader,
                "team": team,
            })

            # --- Team vote ---
            votes = {
                p: strategies[p].vote_team(p, state, rng)
                for p in range(state.n_players)
            }
            state.team_votes = votes
            approved = state.apply_team_vote()
            events.append({
                "type": "team_vote",
                "quest": state.quest_number + 1,
                "votes": votes,
                "approved": approved,
                "team": team,
            })

            if approved:
                break
            # Rejected: new discussion snippet before re-proposal? No — just re-propose.
            # Check if we've run out of quests
            if state.winner is not None:
                break

        if state.winner is not None:
            break

        # --- Quest ---
        quest_team = state.proposed_team or team
        quest_votes = {
            p: strategies[p].quest_action(p, state, rng)
            for p in quest_team
        }
        state.quest_votes = quest_votes
        success = state.apply_quest()
        # quest_number was advanced by apply_quest; use quest_number - 1 for reporting
        completed_quest = state.quest_number  # already incremented
        events.append({
            "type": "quest_result",
            "quest": completed_quest,   # 1-indexed completed quest number
            "success": success,
            "n_fails": sum(1 for v in quest_votes.values() if not v),
            "team": quest_team,
            "votes": quest_votes,
        })

        winner_check = state.check_winner()
        if winner_check is not None:
            if state.phase == AvalonPhase.ASSASSINATION:
                # Assassin tries to identify Merlin
                assassin = state.assassin()
                good_players = state.good_players()
                # Heuristic: guess a Good player (random among good)
                target = rng.choice(good_players)
                state.assassination_target = target
                events.append({
                    "type": "assassination",
                    "assassin": assassin,
                    "target": target,
                    "merlin": state.merlin(),
                    "success": target == state.merlin(),
                })
                if target == state.merlin():
                    state.winner = "evil"
                else:
                    state.winner = "good"
            break

    events.append({"type": "game_outcome", "winner": state.winner})
    return events


# ---------------------------------------------------------------------------
# Token builder
# ---------------------------------------------------------------------------


def _build_tokens(
    events: list[dict],
    state: AvalonState,
    quality_level: str,
) -> list[Token]:
    """Build Token list from event log. Focal player is always player 0."""
    tokens: list[Token] = []
    position = 0
    focal = 0

    # --- Token 0: game setup observation ---
    role = state.roles[focal]
    evil = state.evil_players()
    good = state.good_players()

    if role == AvalonRole.MERLIN:
        evil_str = ", ".join(str(p) for p in sorted(evil))
        setup_text = (
            f"You are Player {focal}. Your role is Merlin. "
            f"You know that Players {evil_str} are Evil. "
            f"Guide the Good team to victory without revealing your identity."
        )
    elif role == AvalonRole.GOOD:
        setup_text = (
            f"You are Player {focal}. Your role is a Good knight. "
            f"Work with your team to complete 3 quests. "
            f"You do not know who is Evil."
        )
    elif role == AvalonRole.EVIL:
        other_evil = [p for p in evil if p != focal]
        other_str = ", ".join(str(p) for p in sorted(other_evil))
        setup_text = (
            f"You are Player {focal}. Your role is Evil. "
            f"Your Evil partner(s): Players {other_str}. "
            f"Sabotage quests while avoiding detection."
        )
    elif role == AvalonRole.ASSASSIN:
        other_evil = [p for p in evil if p != focal]
        other_str = ", ".join(str(p) for p in sorted(other_evil))
        setup_text = (
            f"You are Player {focal}. Your role is the Assassin (Evil). "
            f"Your Evil partner(s): Players {other_str}. "
            f"If Good wins 3 quests, you get one chance to assassinate Merlin."
        )
    else:
        setup_text = f"You are Player {focal}. Your role is {role.value}."

    tokens.append(Token(
        text=setup_text,
        token_ids=[],
        is_observation=True,
        semantic_type="game_setup",
        position=position,
    ))
    position += 1

    # --- Process events ---
    for event in events:
        etype = event["type"]

        if etype == "discussion":
            for (player, stmt) in event["statements"]:
                if player == focal:
                    tokens.append(Token(
                        text=stmt,
                        token_ids=[],
                        is_observation=False,
                        semantic_type="discussion_statement",
                        position=position,
                    ))
                else:
                    tokens.append(Token(
                        text=f"Player {player}: '{stmt}'",
                        token_ids=[],
                        is_observation=True,
                        semantic_type="other_player_statement",
                        position=position,
                    ))
                position += 1

        elif etype == "team_proposal":
            leader = event["leader"]
            team = event["team"]
            team_str = ", ".join(str(p) for p in sorted(team))
            quest_n = event["quest"]
            if leader == focal:
                tokens.append(Token(
                    text=f"I propose team: Players {team_str}.",
                    token_ids=[],
                    is_observation=False,
                    semantic_type="team_proposal",
                    position=position,
                ))
            else:
                tokens.append(Token(
                    text=f"Player {leader} proposes team for Quest {quest_n}: Players {team_str}.",
                    token_ids=[],
                    is_observation=True,
                    semantic_type="team_proposal",
                    position=position,
                ))
            position += 1

        elif etype == "team_vote":
            votes = event["votes"]
            approved = event["approved"]
            team = event["team"]

            # Focal player's vote (action token)
            focal_vote = votes.get(focal)
            if focal_vote is not None:
                vote_word = "Approve" if focal_vote else "Reject"
                tokens.append(Token(
                    text=f"I vote {vote_word}.",
                    token_ids=[],
                    is_observation=False,
                    semantic_type="vote",
                    position=position,
                ))
                position += 1

            # Vote result observation
            vote_parts = ", ".join(
                f"Player {p}: {'Approve' if v else 'Reject'}"
                for p, v in sorted(votes.items())
            )
            result_word = "approved" if approved else "rejected"
            tokens.append(Token(
                text=f"Votes: {vote_parts}. Team {result_word}.",
                token_ids=[],
                is_observation=True,
                semantic_type="vote_result",
                position=position,
            ))
            position += 1

        elif etype == "quest_result":
            quest_votes = event.get("votes", {})
            team = event.get("team", [])
            success = event["success"]
            n_fails = event["n_fails"]
            quest_n = event["quest"]

            # Focal player's quest action (if on team)
            if focal in team:
                focal_quest_vote = quest_votes.get(focal, True)
                action_word = "Pass" if focal_quest_vote else "Fail"
                tokens.append(Token(
                    text=f"I choose to {action_word} this quest.",
                    token_ids=[],
                    is_observation=False,
                    semantic_type="quest_action",
                    position=position,
                ))
                position += 1

            # Quest result observation
            result_word = "succeeded" if success else "failed"
            tokens.append(Token(
                text=f"Quest {quest_n} {result_word}. ({n_fails} fail card(s) played.)",
                token_ids=[],
                is_observation=True,
                semantic_type="quest_result",
                position=position,
            ))
            position += 1

        elif etype == "assassination":
            assassin = event["assassin"]
            target = event["target"]
            merlin = event["merlin"]
            correct = event["success"]
            correct_str = "Correct! Merlin is eliminated." if correct else "Wrong! Merlin survives."
            tokens.append(Token(
                text=(
                    f"The Assassin (Player {assassin}) targets "
                    f"Player {target} as Merlin. {correct_str}"
                ),
                token_ids=[],
                is_observation=True,
                semantic_type="assassination",
                position=position,
            ))
            position += 1

        elif etype == "game_outcome":
            winner = event["winner"] or "unknown"
            tokens.append(Token(
                text=f"The {winner} team wins!",
                token_ids=[],
                is_observation=True,
                semantic_type="game_outcome",
                position=position,
            ))
            position += 1

    return tokens


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class AvalonTrajectoryGenerator:
    """Generates Avalon game trajectories at controlled quality levels."""

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(
        self, quality_level: str, n: int, seed: int
    ) -> list[Trajectory]:
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        rng = random.Random(seed)
        trajectories: list[Trajectory] = []

        for i in range(n):
            traj_seed = rng.randint(0, 2**31)
            traj_rng = random.Random(traj_seed)

            traj = self._generate_one(quality_level, traj_rng)
            trajectories.append(traj)

        return trajectories

    def _generate_one(
        self, quality_level: str, rng: random.Random
    ) -> Trajectory:
        # Assign roles
        roles = _assign_roles(quality_level, rng)

        # Build initial state
        state = AvalonState(
            roles=roles,
            n_players=5,
            phase=AvalonPhase.DISCUSSION,
            quest_number=0,
            current_leader=rng.randint(0, 4),
        )

        # Build strategy map
        focal_strategy = _FOCAL_STRATEGY_MAP[quality_level]()
        strategies: dict[int, AvalonStrategy] = {0: focal_strategy}
        for p in range(1, 5):
            strategies[p] = _make_npc_strategy(p, state)

        # Run game
        events = _run_game(state, strategies, rng)
        state.events = events

        # Build tokens
        tokens = _build_tokens(events, state, quality_level)

        # Quality metrics
        focal_role = state.roles[0]
        is_good_focal = focal_role in (AvalonRole.MERLIN, AvalonRole.GOOD)
        is_evil_focal = focal_role in (AvalonRole.EVIL, AvalonRole.ASSASSIN)

        winner = state.winner
        outcome = 0.0
        if is_good_focal and winner == "good":
            outcome = 1.0
        elif is_evil_focal and winner == "evil":
            outcome = 1.0
        else:
            outcome = -1.0

        total_quests = len(state.quest_results)
        if total_quests > 0:
            if is_good_focal:
                quest_success_rate = state.good_quest_wins() / total_quests
            else:
                quest_success_rate = state.evil_quest_wins() / total_quests
        else:
            quest_success_rate = 0.0

        # team_selection_quality: fraction of focal-led proposals that were good
        focal_proposals = [
            e for e in events
            if e["type"] == "team_proposal" and e["leader"] == 0
        ]
        if focal_proposals:
            good_proposals = 0
            for ep in focal_proposals:
                team = ep["team"]
                if is_good_focal:
                    if not any(p in state.evil_players() for p in team):
                        good_proposals += 1
                else:
                    if any(p in state.evil_players() for p in team):
                        good_proposals += 1
            team_selection_quality = good_proposals / len(focal_proposals)
        else:
            team_selection_quality = 1.0

        # vote_consistency: focal's team votes aligned with their team's interest
        vote_events = [e for e in events if e["type"] == "team_vote"]
        consistent_votes = 0
        total_votes = 0
        for ve in vote_events:
            focal_vote = ve["votes"].get(0)
            if focal_vote is None:
                continue
            team = ve["team"]
            has_evil = any(p in state.evil_players() for p in team)
            if is_good_focal:
                # Good: should reject evil-containing teams
                consistent = (has_evil and not focal_vote) or (not has_evil and focal_vote)
            else:
                # Evil: should approve evil-containing teams
                consistent = (has_evil and focal_vote) or (not has_evil and not focal_vote)
            consistent_votes += int(consistent)
            total_votes += 1
        vote_consistency = consistent_votes / total_votes if total_votes > 0 else 1.0

        # assassination_evasion
        if focal_role == AvalonRole.MERLIN:
            assassinated = state.assassination_target == 0
            assassination_evasion = 0.0 if assassinated else 1.0
        else:
            assassination_evasion = 1.0

        quality_scores = {
            "outcome": outcome,
            "quest_success_rate": quest_success_rate,
            "team_selection_quality": team_selection_quality,
            "vote_consistency": vote_consistency,
            "assassination_evasion": assassination_evasion,
        }

        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            tokens=tokens,
            character_prompt="",
            quality_scores=quality_scores,
            quality_level=quality_level,
            environment="avalon",
            metadata={
                "roles": {str(p): r.value for p, r in roles.items()},
                "focal_role": focal_role.value,
                "winner": winner,
                "quest_results": state.quest_results,
                "events": events,
                "n_quests_attempted": total_quests,
                "assassination_target": state.assassination_target,
            },
        )
