"""Phase 1: Generate Werewolf trajectories at controlled quality levels."""

from __future__ import annotations

import random
import uuid

from ccsm_eval.trajectories.base import Token, Trajectory
from ccsm_eval.trajectories.werewolf.game import WerewolfPhase, WerewolfRole, WerewolfState
from ccsm_eval.trajectories.werewolf.quality import WerewolfQualityScorer
from ccsm_eval.trajectories.werewolf.strategies import (
    BadWerewolfStrategy,
    ChaoticVillagerStrategy,
    CompetentVillagerStrategy,
    ExpertVillagerStrategy,
    ExpertWerewolfStrategy,
    ModerateVillagerStrategy,
    ModerateWerewolfStrategy,
    PassiveVillagerStrategy,
    WerewolfStrategy,
)

QUALITY_LEVELS = [
    "expert_villager",
    "competent_villager",
    "passive_villager",
    "chaotic_villager",
    "expert_werewolf",
    "bad_werewolf",
]

_VILLAGER_LEVELS = {"expert_villager", "competent_villager", "passive_villager", "chaotic_villager"}
_WEREWOLF_LEVELS = {"expert_werewolf", "bad_werewolf"}

_FOCAL_STRATEGY_MAP: dict[str, type] = {
    "expert_villager": ExpertVillagerStrategy,
    "competent_villager": CompetentVillagerStrategy,
    "passive_villager": PassiveVillagerStrategy,
    "chaotic_villager": ChaoticVillagerStrategy,
    "expert_werewolf": ExpertWerewolfStrategy,
    "bad_werewolf": BadWerewolfStrategy,
}


class WerewolfTrajectoryGenerator:
    """Generates Werewolf trajectories at controlled quality levels for CCSM Phase 1."""

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(self, quality_level: str, n: int, seed: int) -> list[Trajectory]:
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        rng = random.Random(seed)
        results = []

        for _ in range(n):
            traj_rng = random.Random(rng.randint(0, 2**31))
            state = self._init_state(quality_level, traj_rng)
            strategies = self._build_strategies(quality_level, state)
            events = self._run_game(state, strategies, traj_rng)
            tokens = self._build_tokens(events, state, quality_level)
            scores = WerewolfQualityScorer().score_from_state(state, focal_player=0)

            traj = Trajectory(
                trajectory_id=str(uuid.uuid4()),
                tokens=tokens,
                character_prompt="",
                quality_scores=scores,
                quality_level=quality_level,
                environment="werewolf",
                metadata={
                    "roles": {p: r.value for p, r in state.roles.items()},
                    "winner": state.winner,
                    "n_days": state.day,
                    "eliminated": state.eliminated,
                    "night_kills": state.night_kills,
                    "focal_player": 0,
                    "focal_role": state.roles[0].value,
                },
            )
            results.append(traj)

        return results

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_state(self, quality_level: str, rng: random.Random) -> WerewolfState:
        """Assign roles and return initial WerewolfState."""
        roles: dict[int, WerewolfRole] = {}

        if quality_level in _VILLAGER_LEVELS:
            # Focal player 0 is a plain Villager
            roles[0] = WerewolfRole.VILLAGER
            # Remaining 4 players (1-4): 1 seer, 1 WW remaining, 2 villagers
            # (so total = 1 focal villager + 2 villagers + 1 seer + 1 WW  = 5 with 1 WW only?
            # Spec says: 1 more WW, 1 seer, 2 villagers for focal-as-villager case)
            npc_roles = [
                WerewolfRole.WEREWOLF,
                WerewolfRole.WEREWOLF,
                WerewolfRole.SEER,
                WerewolfRole.VILLAGER,
            ]
        else:
            # Focal player 0 is a Werewolf
            roles[0] = WerewolfRole.WEREWOLF
            # Remaining 4 players: 1 more WW, 1 seer, 2 villagers
            npc_roles = [
                WerewolfRole.WEREWOLF,
                WerewolfRole.SEER,
                WerewolfRole.VILLAGER,
                WerewolfRole.VILLAGER,
            ]

        npc_ids = [1, 2, 3, 4]
        rng.shuffle(npc_ids)
        for pid, role in zip(npc_ids, npc_roles):
            roles[pid] = role

        return WerewolfState(
            roles=roles,
            alive=set(range(5)),
            phase=WerewolfPhase.NIGHT,
            day=1,
        )

    def _build_strategies(
        self, quality_level: str, state: WerewolfState
    ) -> dict[int, WerewolfStrategy]:
        """Assign strategies to all players."""
        focal_cls = _FOCAL_STRATEGY_MAP[quality_level]
        strategies: dict[int, WerewolfStrategy] = {0: focal_cls()}

        for pid in range(1, 5):
            role = state.roles[pid]
            if role == WerewolfRole.WEREWOLF:
                strategies[pid] = ModerateWerewolfStrategy()
            else:
                strategies[pid] = ModerateVillagerStrategy()

        return strategies

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def _run_game(
        self,
        state: WerewolfState,
        strategies: dict[int, WerewolfStrategy],
        rng: random.Random,
        max_days: int = 6,
    ) -> list[dict]:
        events: list[dict] = []

        while True:
            # ------ Night phase ------
            state.phase = WerewolfPhase.NIGHT

            # WW consensus kill: lowest-id alive WW makes the choice
            alive_ww = sorted(state.werewolves())
            kill_target: int | None = None
            if alive_ww:
                lead_ww = alive_ww[0]
                kill_target = strategies[lead_ww].night_action(lead_ww, state, rng)
                state.pending_kill = kill_target

            # Seer inspection
            seer_id = state.seer()
            seer_event: tuple[int, int, WerewolfRole] | None = None
            if seer_id is not None:
                inspect_target = strategies[seer_id].night_action(seer_id, state, rng)
                if inspect_target is not None:
                    result_role = state.roles[inspect_target]
                    state.seer_knowledge[inspect_target] = result_role
                    seer_event = (seer_id, inspect_target, result_role)

            events.append({
                "phase": "night",
                "day": state.day,
                "kill": kill_target,
                "seer_inspect": seer_event,
            })

            # Apply night kill at start of day
            state.apply_night_kill()

            # Only check village win here (0 WW remaining — can't happen from a kill,
            # but guard for completeness). Never declare WW win before the day vote runs.
            if state.check_winner() == "village":
                events.append({"phase": "game_outcome", "winner": state.winner, "day": state.day})
                break

            # ------ Day phase ------
            state.phase = WerewolfPhase.DAY

            # Discussion — all alive players in sorted order
            for player in sorted(state.alive):
                statement = strategies[player].discuss(player, state, rng)
                state.discussion_log.append((player, statement))
                events.append({
                    "phase": "discussion",
                    "day": state.day,
                    "player": player,
                    "statement": statement,
                })

            # Vote
            votes = {p: strategies[p].vote(p, state, rng) for p in state.alive}
            state.vote_log.append(votes)
            eliminated = state.apply_vote(votes, rng)
            events.append({
                "phase": "vote",
                "day": state.day,
                "votes": votes,
                "eliminated": eliminated,
                "role": state.roles[eliminated],
            })

            if state.check_winner():
                events.append({"phase": "game_outcome", "winner": state.winner, "day": state.day})
                break

            state.day += 1
            if state.day > max_days:
                state.winner = "werewolf"
                state.phase = WerewolfPhase.OVER
                events.append({"phase": "game_outcome", "winner": state.winner, "day": state.day})
                break

        return events

    # ------------------------------------------------------------------
    # Token building
    # ------------------------------------------------------------------

    def _build_tokens(
        self,
        events: list[dict],
        state: WerewolfState,
        quality_level: str,
    ) -> list[Token]:
        tokens: list[Token] = []
        position = 0
        focal_role = state.roles[0]

        def obs(text: str, sem_type: str) -> Token:
            nonlocal position
            t = Token(
                text=text,
                token_ids=[],
                is_observation=True,
                semantic_type=sem_type,
                position=position,
            )
            position += 1
            return t

        def act(text: str, sem_type: str) -> Token:
            nonlocal position
            t = Token(
                text=text,
                token_ids=[],
                is_observation=False,
                semantic_type=sem_type,
                position=position,
            )
            position += 1
            return t

        # 1. Game setup observation
        role_str = focal_role.value.capitalize()
        setup_text = (
            f"You are Player 0. Your role is {role_str}. "
            f"Players: Player 0 (you), Player 1, Player 2, Player 3, Player 4. "
            f"There are 2 Werewolves and 1 Seer among the 5 players."
        )
        tokens.append(obs(setup_text, "game_setup"))

        # 2. Events
        for event in events:
            phase = event["phase"]

            if phase == "night":
                day = event["day"]
                kill = event["kill"]
                seer_inspect = event["seer_inspect"]

                if focal_role == WerewolfRole.WEREWOLF:
                    if kill is not None:
                        text = f"Night {day}: The Werewolves chose Player {kill} as their target."
                    else:
                        text = f"Night {day}: The Werewolves did not choose a target."
                    tokens.append(obs(text, "night_result"))
                elif focal_role == WerewolfRole.SEER and seer_inspect is not None:
                    _, target, result_role = seer_inspect
                    text = (
                        f"Night {day}: You inspected Player {target} — "
                        f"they are a {result_role.value}."
                    )
                    tokens.append(obs(text, "night_result"))
                else:
                    # Villager: learns about kill at start of day
                    if kill is not None:
                        text = f"Night {day}: Player {kill} was eliminated by the Werewolves."
                        tokens.append(obs(text, "night_result"))

            elif phase == "discussion":
                player = event["player"]
                statement = event["statement"]
                if player == 0:
                    tokens.append(act(statement, "focal_statement"))
                else:
                    text = f"Player {player} says: '{statement}'"
                    tokens.append(obs(text, "other_player_statement"))

            elif phase == "vote":
                votes = event["votes"]
                eliminated = event["eliminated"]
                role = event["role"]

                vote_str = ", ".join(
                    f"Player {v}\u2192Player {t}"
                    for v, t in sorted(votes.items())
                )
                vote_text = f"Votes: {vote_str}. Player {eliminated} was eliminated."
                tokens.append(obs(vote_text, "vote_result"))

                reveal_text = f"Player {eliminated} was a {role.value}!"
                tokens.append(obs(reveal_text, "role_reveal"))

            elif phase == "game_outcome":
                winner = event["winner"]
                if winner == "village":
                    reason = "All Werewolves have been eliminated."
                else:
                    reason = "The Werewolves have taken control."
                outcome_text = f"The {winner} team wins! {reason}"
                tokens.append(obs(outcome_text, "game_outcome"))

        return tokens
