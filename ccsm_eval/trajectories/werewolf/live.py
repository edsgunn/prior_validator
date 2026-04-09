"""Phase 2: Live Werewolf environment for LLM agent interaction."""

from __future__ import annotations

import random
import re
from typing import Any

from ccsm_eval.environments.base import TextEnvironment
from ccsm_eval.trajectories.werewolf.game import WerewolfPhase, WerewolfRole, WerewolfState
from ccsm_eval.trajectories.werewolf.quality import WerewolfQualityScorer
from ccsm_eval.trajectories.werewolf.strategies import (
    ModerateVillagerStrategy,
    ModerateWerewolfStrategy,
    WerewolfStrategy,
)

# Agent is always assigned player ID 0 in the live environment.
_AGENT_PLAYER_ID = 0

_PHASE_DISCUSSION = "discussion"
_PHASE_VOTE = "vote"
_PHASE_NIGHT = "night"
_PHASE_OVER = "over"


class WerewolfEnvironment(TextEnvironment):
    """Phase 2 live Werewolf environment for LLM agent interaction.

    The agent plays as Player 0. Scripted NPCs handle all other players.
    The environment steps through Night → Discussion → Vote cycles automatically,
    pausing for agent input during the discussion and vote phases.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng: random.Random = random.Random(seed)
        self._state: WerewolfState | None = None
        self._strategies: dict[int, WerewolfStrategy] = {}
        self._current_phase: str = _PHASE_OVER
        self._pending_observations: list[str] = []
        self._transcript: list[dict[str, Any]] = []
        self._done: bool = True

    # ------------------------------------------------------------------
    # TextEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, str]:
        """Reset the environment and return initial observations for the agent."""
        self._rng = random.Random(self._seed)
        state, strategies = self._assign_roles_and_strategies()
        self._state = state
        self._strategies = strategies
        self._done = False
        self._pending_observations = []
        self._transcript = []

        # Run night phase immediately (agent has no night action choice for now
        # unless they are a Seer — future extension; currently all night actions
        # are auto-resolved and the agent receives the result as an observation).
        night_obs = self._run_night_phase()
        self._current_phase = _PHASE_DISCUSSION

        # Build initial observation
        focal_role = state.roles[_AGENT_PLAYER_ID]
        role_text = focal_role.value.capitalize()

        intro_lines = [
            f"Welcome to Werewolf. You are Player {_AGENT_PLAYER_ID}. Your role is {role_text}.",
            "Players: Player 0 (you), Player 1, Player 2, Player 3, Player 4.",
            "There are 2 Werewolves and 1 Seer among the 5 players.",
            "Each night the Werewolves eliminate a player. Each day all players discuss "
            "and vote to eliminate a suspect.",
            "Village wins by eliminating all Werewolves. Werewolves win when they equal "
            "or outnumber the remaining villagers.",
        ]

        if focal_role == WerewolfRole.WEREWOLF:
            ww_ids = [p for p in state.alive if state.roles[p] == WerewolfRole.WEREWOLF and p != _AGENT_PLAYER_ID]
            if ww_ids:
                intro_lines.append(f"Your fellow Werewolf is Player {ww_ids[0]}.")

        intro_obs = "\n".join(intro_lines)

        if night_obs:
            intro_obs += "\n\n" + night_obs

        intro_obs += (
            f"\n\nDay {state.day} — Discussion phase. "
            "Please share your thoughts with the group."
        )

        return {"agent": intro_obs}

    def current_agent(self) -> str:
        """Return 'agent' when it's the agent's turn, 'env' otherwise."""
        if self._done:
            return "env"
        if self._current_phase in (_PHASE_DISCUSSION, _PHASE_VOTE):
            return "agent"
        return "env"

    def step(self, agent_id: str, action: str) -> dict:
        """Process the agent's action and advance the environment.

        The agent provides:
          - A discussion statement during the discussion phase.
          - A vote (mentioning "Player N") during the vote phase.

        NPCs act automatically before and after the agent's turn.
        """
        if self._done:
            return {"done": True, "info": self._build_info()}

        state = self._state
        observations: list[str] = []

        if self._current_phase == _PHASE_DISCUSSION:
            obs = self._run_discussion_phase(agent_statement=action)
            observations.extend(obs)
            self._current_phase = _PHASE_VOTE
            observations.append(
                f"Day {state.day} — Vote phase. "
                "Please cast your vote. State which player you vote to eliminate "
                "(e.g., 'I vote for Player 2' or simply 'Player 2')."
            )

        elif self._current_phase == _PHASE_VOTE:
            obs, game_ended = self._run_vote_phase(agent_vote_action=action)
            observations.extend(obs)

            if game_ended or self._done:
                pass
            else:
                # Run night phase and start new day
                night_obs = self._run_night_phase()
                if night_obs:
                    observations.append(night_obs)

                if not self._done:
                    self._current_phase = _PHASE_DISCUSSION
                    observations.append(
                        f"Day {state.day} — Discussion phase. "
                        "Please share your thoughts with the group."
                    )
        else:
            # Unexpected phase — return current state
            pass

        obs_text = "\n\n".join(observations) if observations else ""
        result: dict[str, Any] = {
            "agent": obs_text,
            "done": self._done,
            "info": self._build_info() if self._done else {},
        }
        return result

    # ------------------------------------------------------------------
    # Internal phase runners
    # ------------------------------------------------------------------

    def _run_night_phase(self) -> str:
        """Run night phase (WW kill + seer inspect). Return observation text for agent."""
        state = self._state
        focal_role = state.roles[_AGENT_PLAYER_ID]
        lines: list[str] = []

        # WW consensus kill: lowest-id alive WW
        alive_ww = sorted(state.werewolves())
        kill_target: int | None = None
        if alive_ww:
            lead_ww = alive_ww[0]
            if lead_ww == _AGENT_PLAYER_ID:
                # Agent is the lead WW — auto-pick using moderate WW strategy for now
                # (live WW night action is future extension; for now auto-resolve)
                strategy = ModerateWerewolfStrategy()
                kill_target = strategy.night_action(lead_ww, state, self._rng)
            else:
                kill_target = self._strategies[lead_ww].night_action(lead_ww, state, self._rng)
            state.pending_kill = kill_target

        # Seer inspection (NPC seer only)
        seer_id = state.seer()
        if seer_id is not None and seer_id != _AGENT_PLAYER_ID:
            inspect_target = self._strategies[seer_id].night_action(seer_id, state, self._rng)
            if inspect_target is not None:
                state.seer_knowledge[inspect_target] = state.roles[inspect_target]
        elif seer_id == _AGENT_PLAYER_ID:
            # Agent is seer — auto-inspect random for now (future: ask agent)
            others = [p for p in state.alive if p != _AGENT_PLAYER_ID and p not in state.seer_knowledge]
            if others:
                target = self._rng.choice(others)
                result_role = state.roles[target]
                state.seer_knowledge[target] = result_role
                lines.append(
                    f"Night {state.day}: You used your Seer ability and inspected "
                    f"Player {target} — they are a {result_role.value}."
                )

        # Apply kill
        killed = state.apply_night_kill()

        # Build observation for agent
        if focal_role == WerewolfRole.WEREWOLF:
            if kill_target is not None:
                lines.append(
                    f"Night {state.day}: The Werewolves chose Player {kill_target} as their target."
                )
        else:
            if killed is not None:
                lines.append(
                    f"Night {state.day}: Player {killed} was found dead. "
                    f"They were eliminated by the Werewolves."
                )
            else:
                lines.append(f"Night {state.day}: No one was eliminated during the night.")

        # Only check for village win after night kill (WW win requires a day vote first).
        if state.check_winner() == "village":
            self._done = True
            self._current_phase = _PHASE_OVER
            lines.append(self._winner_text(state.winner))

        self._transcript.append({
            "phase": "night",
            "day": state.day - 1 if state.day > 1 else state.day,
            "kill": killed,
        })

        return "\n".join(lines) if lines else ""

    def _run_discussion_phase(self, agent_statement: str) -> list[str]:
        """Run discussion: NPCs before agent (lower IDs), agent, then NPCs after."""
        state = self._state
        observations: list[str] = []

        # NPCs with player_id < 0 go first (none, since agent is player 0)
        # Agent goes first among sorted alive (player 0)
        # Then remaining NPCs in sorted order

        # Record agent's statement
        state.discussion_log.append((_AGENT_PLAYER_ID, agent_statement))
        self._transcript.append({
            "phase": "discussion",
            "day": state.day,
            "player": _AGENT_PLAYER_ID,
            "statement": agent_statement,
        })

        # NPCs discuss in sorted order (all have IDs > 0)
        for pid in sorted(state.alive):
            if pid == _AGENT_PLAYER_ID:
                continue
            statement = self._strategies[pid].discuss(pid, state, self._rng)
            state.discussion_log.append((pid, statement))
            self._transcript.append({
                "phase": "discussion",
                "day": state.day,
                "player": pid,
                "statement": statement,
            })
            observations.append(f"Player {pid} says: '{statement}'")

        return observations

    def _run_vote_phase(self, agent_vote_action: str) -> tuple[list[str], bool]:
        """Parse agent vote, run NPC votes, apply result. Return observations and done flag."""
        state = self._state
        observations: list[str] = []

        valid_targets = [p for p in state.alive if p != _AGENT_PLAYER_ID]
        agent_vote = self._parse_vote(agent_vote_action, valid_targets)

        if agent_vote is None:
            # Invalid vote — assign random valid target
            agent_vote = self._rng.choice(valid_targets) if valid_targets else valid_targets[0]
            observations.append(
                f"(Could not parse your vote. Randomly assigning vote to Player {agent_vote}.)"
            )

        # Collect all votes
        votes: dict[int, int] = {_AGENT_PLAYER_ID: agent_vote}
        for pid in state.alive:
            if pid == _AGENT_PLAYER_ID:
                continue
            votes[pid] = self._strategies[pid].vote(pid, state, self._rng)

        state.vote_log.append(votes)
        eliminated = state.apply_vote(votes, self._rng)
        eliminated_role = state.roles[eliminated]

        vote_str = ", ".join(
            f"Player {v}\u2192Player {t}" for v, t in sorted(votes.items())
        )
        observations.append(
            f"Vote results: {vote_str}. "
            f"Player {eliminated} was eliminated."
        )
        observations.append(f"Player {eliminated} was a {eliminated_role.value}!")

        self._transcript.append({
            "phase": "vote",
            "day": state.day,
            "votes": dict(votes),
            "eliminated": eliminated,
            "role": eliminated_role.value,
        })

        if state.check_winner():
            self._done = True
            self._current_phase = _PHASE_OVER
            observations.append(self._winner_text(state.winner))
            return observations, True

        state.day += 1
        return observations, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assign_roles_and_strategies(
        self,
    ) -> tuple[WerewolfState, dict[int, WerewolfStrategy]]:
        """Randomly assign roles and build strategies."""
        all_roles = [
            WerewolfRole.WEREWOLF,
            WerewolfRole.WEREWOLF,
            WerewolfRole.SEER,
            WerewolfRole.VILLAGER,
            WerewolfRole.VILLAGER,
        ]
        self._rng.shuffle(all_roles)
        roles: dict[int, WerewolfRole] = {i: all_roles[i] for i in range(5)}

        state = WerewolfState(
            roles=roles,
            alive=set(range(5)),
            phase=WerewolfPhase.NIGHT,
            day=1,
        )

        strategies: dict[int, WerewolfStrategy] = {}
        for pid in range(5):
            if pid == _AGENT_PLAYER_ID:
                # Agent handles their own actions via step()
                strategies[pid] = ModerateVillagerStrategy()  # fallback, not normally called
            elif roles[pid] == WerewolfRole.WEREWOLF:
                strategies[pid] = ModerateWerewolfStrategy()
            else:
                strategies[pid] = ModerateVillagerStrategy()

        return state, strategies

    def _parse_vote(self, action: str, valid_targets: list[int]) -> int | None:
        """Extract a player ID vote from free-form text."""
        # Try "Player N" pattern first
        matches = re.findall(r'[Pp]layer\s+(\d+)', action)
        for m in matches:
            pid = int(m)
            if pid in valid_targets:
                return pid

        # Try bare number
        nums = re.findall(r'\b(\d+)\b', action)
        for n in nums:
            pid = int(n)
            if pid in valid_targets:
                return pid

        return None

    def _winner_text(self, winner: str | None) -> str:
        if winner == "village":
            return "Game over! The village team wins — all Werewolves have been eliminated!"
        elif winner == "werewolf":
            return "Game over! The Werewolves win — they have taken control of the village!"
        return "Game over!"

    def _build_info(self) -> dict[str, Any]:
        """Build end-of-game info dict."""
        if self._state is None:
            return {}

        state = self._state
        scorer = WerewolfQualityScorer()
        scores = scorer.score_from_state(state, focal_player=_AGENT_PLAYER_ID)

        focal_role = state.roles[_AGENT_PLAYER_ID]
        survived = (
            _AGENT_PLAYER_ID not in state.eliminated
            and _AGENT_PLAYER_ID not in state.night_kills
        )

        return {
            "outcome": state.winner,
            "survival": survived,
            "vote_accuracy": scores.get("vote_accuracy", 0.0),
            "deception_success": scores.get("deception_success", 1.0),
            "agent_role": focal_role.value,
            "n_days": state.day,
            "eliminated_order": state.eliminated,
            "night_kills": state.night_kills,
            "transcript": self._transcript,
            "quality_scores": scores,
        }
