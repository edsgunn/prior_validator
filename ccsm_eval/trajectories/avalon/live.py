"""Phase 2 live Avalon environment for LLM agent interaction.

The agent plays as one of the 5 players. Scripted NPCs fill the remaining roles.
The environment handles all game mechanics and NPC actions internally.
"""

from __future__ import annotations

import random
import re
from typing import Optional

from ccsm_eval.environments.base import TextEnvironment
from ccsm_eval.trajectories.avalon.game import (
    AvalonPhase,
    AvalonRole,
    AvalonState,
    QUEST_TEAM_SIZES,
)
from ccsm_eval.trajectories.avalon.strategies import (
    AvalonStrategy,
    MerlinStrategy,
    ModerateEvilStrategy,
    ModerateGoodStrategy,
)


# ---------------------------------------------------------------------------
# NPC strategy factory
# ---------------------------------------------------------------------------


def _make_npc(role: AvalonRole) -> AvalonStrategy:
    if role == AvalonRole.MERLIN:
        return MerlinStrategy()
    elif role in (AvalonRole.EVIL, AvalonRole.ASSASSIN):
        return ModerateEvilStrategy()
    return ModerateGoodStrategy()


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------


def _assign_roles_random(rng: random.Random) -> dict[int, AvalonRole]:
    """Randomly assign roles to 5 players (player 0 = agent)."""
    all_roles = [
        AvalonRole.MERLIN,
        AvalonRole.GOOD,
        AvalonRole.GOOD,
        AvalonRole.EVIL,
        AvalonRole.ASSASSIN,
    ]
    rng.shuffle(all_roles)
    return {i: all_roles[i] for i in range(5)}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_PLAYER_LIST_RE = re.compile(r"[Pp]layer[s]?\s*([\d,\s]+)")
_SINGLE_PLAYER_RE = re.compile(r"\b(\d)\b")


def _parse_team_proposal(text: str, size: int, n_players: int = 5) -> Optional[list[int]]:
    """Extract a list of player IDs from agent text."""
    m = _PLAYER_LIST_RE.search(text)
    if m:
        nums_str = m.group(1)
        nums = re.findall(r"\d+", nums_str)
        players = [int(n) for n in nums if 0 <= int(n) < n_players]
        players = list(dict.fromkeys(players))  # deduplicate, preserve order
        if players:
            return players[:size]
    # Try to extract any numbers in range
    nums = re.findall(r"\b(\d)\b", text)
    players = list(dict.fromkeys(int(n) for n in nums if 0 <= int(n) < n_players))
    if players:
        return players[:size]
    return None


def _parse_vote(text: str) -> Optional[bool]:
    """Parse 'approve' -> True, 'reject' -> False."""
    lower = text.lower()
    if "approve" in lower:
        return True
    if "reject" in lower:
        return False
    return None


def _parse_quest_action(text: str) -> Optional[bool]:
    """Parse 'pass' -> True (pass), 'fail' -> False (fail)."""
    lower = text.lower()
    if "pass" in lower:
        return True
    if "fail" in lower:
        return False
    return None


def _parse_assassination(text: str, n_players: int = 5) -> Optional[int]:
    """Parse 'I accuse Player X' or 'Player X' -> X."""
    m = re.search(r"[Pp]layer\s*(\d+)", text)
    if m:
        p = int(m.group(1))
        if 0 <= p < n_players:
            return p
    return None


# ---------------------------------------------------------------------------
# AvalonEnvironment
# ---------------------------------------------------------------------------


class AvalonEnvironment(TextEnvironment):
    """Live 5-player Avalon environment.

    The agent (player 0) plays against 4 scripted NPC players. The environment
    drives NPC decisions internally and only surfaces agent decision points.

    Agent action points:
        1. Discussion — provide a free-text statement each quest round.
        2. Team proposal — if agent is current leader.
        3. Team vote — approve or reject proposed team.
        4. Quest action — pass or fail (only on quest team).
        5. Assassination — if agent is the Assassin and Good wins 3 quests.
    """

    _AGENT_ID = "agent"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

        # Set by reset()
        self._state: Optional[AvalonState] = None
        self._npc_strategies: dict[int, AvalonStrategy] = {}
        self._done: bool = True
        self._pending_phase: Optional[str] = None  # which sub-phase needs agent action
        self._last_obs: str = ""
        self._discussed_quest: int = -1  # tracks which quest number has had discussion

    # ------------------------------------------------------------------
    # TextEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, str]:
        """Set up a new game and return initial role description."""
        roles = _assign_roles_random(self._rng)
        self._state = AvalonState(
            roles=roles,
            n_players=5,
            phase=AvalonPhase.DISCUSSION,
            quest_number=0,
            current_leader=self._rng.randint(0, 4),
        )
        self._done = False
        self._discussed_quest = -1
        self._npc_strategies = {
            p: _make_npc(roles[p])
            for p in range(1, 5)
        }

        # Build role description for agent (player 0)
        obs = self._build_role_description()

        # Advance to the first agent decision point
        intro_obs = self._advance_to_agent_turn()
        if intro_obs:
            obs = obs + "\n\n" + intro_obs

        return {self._AGENT_ID: obs}

    def current_agent(self) -> str:
        return self._AGENT_ID

    def step(self, agent_id: str, action: str) -> dict:
        """Process the agent's action and advance to the next agent decision point."""
        if self._done:
            return {
                self._AGENT_ID: "The game is already over.",
                "done": True,
                "info": {},
            }

        state = self._state
        pending = self._pending_phase

        # --- Handle agent action based on pending phase ---
        if pending == "discussion":
            # Free text statement — any text is valid
            statement = action.strip() if action.strip() else "No comment."
            state.discussion_log.append((0, statement))
            obs = self._advance_to_agent_turn()
            return self._make_response(obs)

        elif pending == "team_proposal":
            size = state.current_team_size()
            team = _parse_team_proposal(action, size)
            if team is None or len(team) == 0:
                # Random fallback
                all_p = list(range(5))
                self._rng.shuffle(all_p)
                team = sorted(all_p[:size])

            # Pad/trim to correct size
            team = list(dict.fromkeys(team))
            while len(team) < size:
                for p in range(5):
                    if p not in team:
                        team.append(p)
                    if len(team) >= size:
                        break
            team = team[:size]
            state.proposed_team = team
            obs = f"You proposed team: Players {', '.join(str(p) for p in sorted(team))}.\n"
            obs += self._advance_to_agent_turn()
            return self._make_response(obs)

        elif pending == "team_vote":
            vote = _parse_vote(action)
            if vote is None:
                vote = True  # default Approve
            state.team_votes[0] = vote

            # Collect NPC votes
            for p in range(1, 5):
                state.team_votes[p] = self._npc_strategies[p].vote_team(p, state, self._rng)

            # Snapshot votes before apply_team_vote resets them
            votes_snapshot: dict[int, bool] = dict(state.team_votes)
            approved = state.apply_team_vote()

            vote_word = "Approve" if vote else "Reject"
            result_word = "approved" if approved else "rejected"
            vote_summary = ", ".join(
                f"Player {p}: {'Approve' if v else 'Reject'}"
                for p, v in sorted(votes_snapshot.items())
            )
            obs = (
                f"You voted: {vote_word}.\n"
                f"All votes — {vote_summary}.\n"
                f"Team {result_word}.\n"
            )
            obs += self._advance_to_agent_turn()
            return self._make_response(obs)

        elif pending == "quest_action":
            qa = _parse_quest_action(action)
            role = state.roles.get(0)
            if qa is None:
                qa = True  # default Pass (safe)

            # Collect all quest votes for players on team
            quest_team = state.proposed_team or []
            quest_votes: dict[int, bool] = {}
            if 0 in quest_team:
                quest_votes[0] = qa
            for p in quest_team:
                if p != 0:
                    quest_votes[p] = self._npc_strategies[p].quest_action(p, state, self._rng)

            state.quest_votes = quest_votes
            success = state.apply_quest()
            n_fails = sum(1 for v in quest_votes.values() if not v)
            quest_n = state.quest_number  # already incremented by apply_quest
            result_word = "succeeded" if success else "failed"
            action_word = "Pass" if qa else "Fail"
            obs = (
                f"You chose to {action_word} this quest.\n"
                f"Quest {quest_n} {result_word}. ({n_fails} fail card(s) played.)\n"
            )

            winner_check = state.check_winner()
            if winner_check is not None:
                if state.phase == AvalonPhase.ASSASSINATION:
                    obs += self._advance_to_agent_turn()
                    return self._make_response(obs)
                # Game over
                self._done = True
                obs += f"The {state.winner} team wins!\n"
                return self._make_terminal_response(obs)

            obs += self._advance_to_agent_turn()
            return self._make_response(obs)

        elif pending == "assassination":
            target = _parse_assassination(action)
            if target is None:
                # Random guess among Good players
                good = state.good_players()
                target = self._rng.choice(good)

            merlin = state.merlin()
            correct = target == merlin
            state.assassination_target = target
            correct_str = "Correct! Merlin is eliminated." if correct else "Wrong! Merlin survives."
            obs = (
                f"You (Assassin) target Player {target} as Merlin. {correct_str}\n"
            )
            if correct:
                state.winner = "evil"
            else:
                state.winner = "good"
            state.phase = AvalonPhase.OVER
            self._done = True
            obs += f"The {state.winner} team wins!\n"
            return self._make_terminal_response(obs)

        else:
            return {
                self._AGENT_ID: f"Unexpected pending phase: {pending!r}",
                "done": self._done,
                "info": {},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_role_description(self) -> str:
        """Build initial role description for the agent."""
        state = self._state
        role = state.roles[0]
        evil = state.evil_players()
        good = state.good_players()

        if role == AvalonRole.MERLIN:
            evil_str = ", ".join(str(p) for p in sorted(evil))
            return (
                f"You are Player 0. Your role is Merlin.\n"
                f"You know that Players {evil_str} are Evil.\n"
                f"Guide the Good team to complete 3 quests. "
                f"Do not reveal your identity — the Assassin will try to find you.\n"
                f"Quest team sizes for 5 players: {QUEST_TEAM_SIZES}."
            )
        elif role == AvalonRole.GOOD:
            return (
                f"You are Player 0. Your role is a Good knight.\n"
                f"Work with your team to complete 3 quests. "
                f"You do not know who is Evil.\n"
                f"Quest team sizes for 5 players: {QUEST_TEAM_SIZES}."
            )
        elif role == AvalonRole.EVIL:
            other_evil = [p for p in evil if p != 0]
            other_str = ", ".join(str(p) for p in sorted(other_evil))
            return (
                f"You are Player 0. Your role is Evil.\n"
                f"Your Evil partner(s): Players {other_str}.\n"
                f"Sabotage quests while blending in. Prevent Good from completing 3 quests.\n"
                f"Quest team sizes for 5 players: {QUEST_TEAM_SIZES}."
            )
        elif role == AvalonRole.ASSASSIN:
            other_evil = [p for p in evil if p != 0]
            other_str = ", ".join(str(p) for p in sorted(other_evil))
            return (
                f"You are Player 0. Your role is the Assassin (Evil).\n"
                f"Your Evil partner(s): Players {other_str}.\n"
                f"Sabotage quests while blending in. "
                f"If Good wins 3 quests, you may assassinate who you believe is Merlin.\n"
                f"Quest team sizes for 5 players: {QUEST_TEAM_SIZES}."
            )
        return f"You are Player 0. Your role is {role.value}."

    def _advance_to_agent_turn(self) -> str:
        """Run NPC actions until it's the agent's turn. Return observation text.

        Phase flow per quest round:
          DISCUSSION -> (agent speaks) -> TEAM_PROPOSAL -> (leader proposes)
          -> TEAM_VOTE -> (all vote) -> QUEST or back to DISCUSSION
          -> QUEST -> (quest resolves) -> DISCUSSION (next round) or end
        """
        state = self._state
        obs_parts: list[str] = []

        for _ in range(200):  # safety limit to prevent infinite loops
            if self._done:
                break

            if state.winner is not None:
                self._done = True
                obs_parts.append(f"The {state.winner} team wins!\n")
                break

            if state.quest_number >= 5:
                self._done = True
                break

            phase = state.phase

            # --- Discussion ---
            if phase == AvalonPhase.DISCUSSION:
                quest_n = state.quest_number
                if self._discussed_quest == quest_n:
                    # Already had discussion for this quest (e.g. after team rejection)
                    # Skip to TEAM_PROPOSAL directly
                    state.phase = AvalonPhase.TEAM_PROPOSAL
                    continue

                # NPCs discuss first, then agent
                self._discussed_quest = quest_n
                npc_stmts: list[str] = []
                for p in range(1, 5):
                    stmt = self._npc_strategies[p].discuss(p, state, self._rng)
                    state.discussion_log.append((p, stmt))
                    npc_stmts.append(f"Player {p}: \"{stmt}\"")
                if npc_stmts:
                    obs_parts.append(
                        f"--- Quest {state.quest_number + 1} Discussion ---\n"
                        + "\n".join(npc_stmts) + "\n"
                    )
                # Transition: after discussion, move to TEAM_PROPOSAL
                state.phase = AvalonPhase.TEAM_PROPOSAL
                # Agent needs to provide discussion statement
                self._pending_phase = "discussion"
                obs_parts.append("It is your turn to speak. Provide your discussion statement.")
                return "".join(obs_parts)

            # --- Team Proposal ---
            elif phase == AvalonPhase.TEAM_PROPOSAL:
                leader = state.current_leader
                size = state.current_team_size()
                if leader == 0:
                    # Agent proposes
                    self._pending_phase = "team_proposal"
                    obs_parts.append(
                        f"You are the leader for Quest {state.quest_number + 1}. "
                        f"Propose a team of {size} players. "
                        f"Format: 'I propose Players 1, 3' or 'My team: 2 and 4'."
                    )
                    return "".join(obs_parts)
                else:
                    # NPC proposes
                    team = self._npc_strategies[leader].propose_team(leader, state, self._rng)
                    team = list(dict.fromkeys(team))
                    while len(team) < size:
                        for p in range(5):
                            if p not in team:
                                team.append(p)
                            if len(team) >= size:
                                break
                    team = team[:size]
                    state.proposed_team = team
                    obs_parts.append(
                        f"Player {leader} proposes team for Quest {state.quest_number + 1}: "
                        f"Players {', '.join(str(p) for p in sorted(team))}.\n"
                    )
                    state.phase = AvalonPhase.TEAM_VOTE
                    # Continue to TEAM_VOTE handling in next iteration

            # --- Team Vote ---
            elif phase == AvalonPhase.TEAM_VOTE:
                team = state.proposed_team or []
                team_str = ", ".join(str(p) for p in sorted(team))
                self._pending_phase = "team_vote"
                obs_parts.append(
                    f"Vote on proposed team: Players {team_str}.\n"
                    f"Say 'I vote Approve' or 'I vote Reject'."
                )
                return "".join(obs_parts)

            # --- Quest ---
            elif phase == AvalonPhase.QUEST:
                team = state.proposed_team or []
                if 0 in team:
                    role = state.roles[0]
                    self._pending_phase = "quest_action"
                    if role in (AvalonRole.EVIL, AvalonRole.ASSASSIN):
                        prompt = (
                            f"You are on the quest team: Players {', '.join(str(p) for p in sorted(team))}.\n"
                            f"Say 'I choose to Pass' or 'I choose to Fail' this quest."
                        )
                    else:
                        prompt = (
                            f"You are on the quest team: Players {', '.join(str(p) for p in sorted(team))}.\n"
                            f"Say 'I choose to Pass' this quest."
                        )
                    obs_parts.append(prompt)
                    return "".join(obs_parts)
                else:
                    # Agent not on team; run quest with NPCs only
                    quest_votes: dict[int, bool] = {}
                    for p in team:
                        if p in self._npc_strategies:
                            quest_votes[p] = self._npc_strategies[p].quest_action(p, state, self._rng)
                        else:
                            quest_votes[p] = True  # default Pass for unknown
                    state.quest_votes = quest_votes
                    success = state.apply_quest()
                    n_fails = sum(1 for v in quest_votes.values() if not v)
                    quest_n = state.quest_number  # already incremented by apply_quest
                    result_word = "succeeded" if success else "failed"
                    obs_parts.append(
                        f"Quest {quest_n} {result_word}. ({n_fails} fail card(s) played.)\n"
                    )

                    winner_check = state.check_winner()
                    if winner_check is not None:
                        if state.phase == AvalonPhase.ASSASSINATION:
                            continue  # handled in next iteration
                        self._done = True
                        obs_parts.append(f"The {state.winner} team wins!\n")
                        break
                    # apply_quest sets phase to DISCUSSION; loop continues

            # --- Assassination ---
            elif phase == AvalonPhase.ASSASSINATION:
                assassin_player = state.assassin()
                if assassin_player == 0:
                    self._pending_phase = "assassination"
                    obs_parts.append(
                        f"Good has won 3 quests. As the Assassin, you have one chance "
                        f"to identify and assassinate Merlin.\n"
                        f"Say 'I accuse Player X as Merlin' (replace X with your guess)."
                    )
                    return "".join(obs_parts)
                else:
                    # NPC Assassin guesses randomly among Good players
                    good = state.good_players()
                    target = self._rng.choice(good)
                    state.assassination_target = target
                    merlin = state.merlin()
                    correct = target == merlin
                    correct_str = (
                        "Correct! Merlin is eliminated." if correct else "Wrong! Merlin survives."
                    )
                    state.winner = "evil" if correct else "good"
                    state.phase = AvalonPhase.OVER
                    self._done = True
                    obs_parts.append(
                        f"The Assassin (Player {assassin_player}) targets "
                        f"Player {target} as Merlin. {correct_str}\n"
                        f"The {state.winner} team wins!\n"
                    )
                    break

            elif phase == AvalonPhase.OVER:
                self._done = True
                break

        return "".join(obs_parts)

    def _make_response(self, obs: str) -> dict:
        """Build a step response dict."""
        if self._done:
            return self._make_terminal_response(obs)
        return {
            self._AGENT_ID: obs,
            "done": False,
            "info": {},
        }

    def _make_terminal_response(self, obs: str) -> dict:
        """Build terminal step response with full info."""
        state = self._state
        return {
            self._AGENT_ID: obs,
            "done": True,
            "info": {
                "outcome": state.winner,
                "quest_results": state.quest_results,
                "agent_role": state.roles.get(0, AvalonRole.GOOD).value,
                "assassination_target": state.assassination_target,
            },
        }
