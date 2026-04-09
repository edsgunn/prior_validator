"""Counterfactual edit generation for Werewolf trajectories."""

from __future__ import annotations

import random

from ccsm_eval.environments.base import CounterfactualEditor
from ccsm_eval.trajectories.base import CounterfactualEdit, Token, Trajectory
from ccsm_eval.trajectories.werewolf.game import WerewolfRole


class WerewolfCounterfactualEditor(CounterfactualEditor):
    """Generates counterfactual edits for Werewolf focal-agent statements."""

    def sample_edit_positions(self, trajectory: Trajectory) -> list[int]:
        """Return indices of focal_statement action tokens eligible for editing.

        With >= 3 actions: skip first and last (avoid edge effects).
        With 2 actions: return just the first (has a following vote round).
        With 1 action: return it if it has subsequent observation tokens.
        """
        action_positions = [
            i
            for i, tok in enumerate(trajectory.tokens)
            if not tok.is_observation and tok.semantic_type == "focal_statement"
        ]
        if not action_positions:
            return []
        if len(action_positions) >= 3:
            return action_positions[1:-1]
        # 1-2 actions: return all except the very last (no following round to measure)
        return action_positions[:-1] if len(action_positions) > 1 else action_positions

    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        """Replace a focal_statement action token with a better or worse alternative.

        Args:
            trajectory: The original trajectory.
            position: Index into trajectory.tokens of the action token to replace.
            direction: "up" for a better statement, "down" for a worse statement.

        Returns:
            A CounterfactualEdit with the replacement statement.
        """
        original_token = trajectory.tokens[position]
        if original_token.is_observation or original_token.semantic_type != "focal_statement":
            raise ValueError(
                f"Token at position {position} is not a focal_statement action token."
            )

        metadata = trajectory.metadata
        focal_role = metadata.get("focal_role", "villager")
        roles: dict[str, str] = metadata.get("roles", {})

        # Convert string keys to int
        int_roles = {int(k): v for k, v in roles.items()}

        # Identify werewolves and villagers from metadata
        ww_players = [p for p, r in int_roles.items() if r == WerewolfRole.WEREWOLF.value and p != 0]
        villager_players = [
            p for p, r in int_roles.items()
            if r in (WerewolfRole.VILLAGER.value, WerewolfRole.SEER.value) and p != 0
        ]

        rng = random.Random(hash((trajectory.trajectory_id, position, direction)))

        if direction == "up":
            replacement_text = self._generate_up_statement(
                focal_role, ww_players, villager_players, rng
            )
            quality_delta = 0.2
        elif direction == "down":
            replacement_text = self._generate_down_statement(
                focal_role, ww_players, villager_players, rng
            )
            quality_delta = -0.2
        else:
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        # Find the observation tokens immediately following this action token
        # (vote_result and role_reveal for the same round, if present)
        following_obs_tokens: list[Token] = []
        for tok in trajectory.tokens[position + 1:]:
            if not tok.is_observation:
                break
            if tok.semantic_type in ("vote_result", "role_reveal"):
                following_obs_tokens.append(tok)
            else:
                break

        # Since NPC votes are unchanged and vote outcome depends on all players,
        # we keep the following observation tokens identical for both original and replacement.
        return CounterfactualEdit(
            trajectory_id=trajectory.trajectory_id,
            edit_position=position,
            original_action=original_token.text,
            replacement_action=replacement_text,
            direction=direction,
            quality_delta=quality_delta,
            original_tokens=list(following_obs_tokens),
            replacement_tokens=list(following_obs_tokens),
        )

    # ------------------------------------------------------------------
    # Statement generation helpers
    # ------------------------------------------------------------------

    def _generate_up_statement(
        self,
        focal_role: str,
        ww_players: list[int],
        villager_players: list[int],
        rng: random.Random,
    ) -> str:
        """Generate a higher-quality statement."""
        if focal_role == WerewolfRole.WEREWOLF.value:
            # Better WW statement: deflect convincingly toward a villager
            if villager_players:
                target = rng.choice(villager_players)
                return (
                    f"I've been watching Player {target} very carefully. "
                    f"Their behaviour throughout the game strongly suggests they are a Werewolf — "
                    f"they deflect every time suspicion arises."
                )
            return "I've been analysing everyone's behaviour carefully and I'm confident we can identify the Werewolves."
        else:
            # Better Villager/Seer statement: name an actual werewolf
            if ww_players:
                target = rng.choice(ww_players)
                return (
                    f"Based on my investigation, Player {target} is a confirmed Werewolf. "
                    f"We must vote them out immediately."
                )
            return "I've been paying close attention to voting patterns and I believe I know who the Werewolves are."

    def _generate_down_statement(
        self,
        focal_role: str,
        ww_players: list[int],
        villager_players: list[int],
        rng: random.Random,
    ) -> str:
        """Generate a lower-quality statement."""
        if focal_role == WerewolfRole.WEREWOLF.value:
            # Worse WW statement: suspicious, self-contradictory
            candidates = villager_players or [p for p in range(1, 5)]
            if len(candidates) >= 2:
                picks = rng.sample(candidates, 2)
                return (
                    f"Player {picks[0]} seems suspicious... but actually maybe it's "
                    f"Player {picks[1]}. I'm not sure what to think."
                )
            return "I definitely wasn't doing anything suspicious last night. Not that anything happened."
        else:
            # Worse Villager/Seer statement: accuse an actual villager (false accusation)
            if villager_players:
                target = rng.choice(villager_players)
                return f"I'm certain Player {target} is a Werewolf. We should vote them out now."
            return "I don't really have any information. I'm not sure who to suspect."
