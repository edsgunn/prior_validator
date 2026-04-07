"""Negotiation trajectory generator.

Produces templated (Tier 1) and natural language (Tier 2) negotiation games
across different quality levels. Player B uses a fixed tit-for-tat strategy.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from ccsm_eval.trajectories.base import Token, Trajectory
from ccsm_eval.trajectories.negotiation.strategies import (
    QUALITY_LEVELS,
    STRATEGY_REGISTRY,
    ItemPool,
    NegotiationAction,
    Offer,
    OptimalStrategy,
    TitForTatStrategy,
    Valuations,
)


# ---------------------------------------------------------------------------
# Default scenario configurations
# ---------------------------------------------------------------------------

# Each scenario: (item_counts, player_a_values, player_b_values)
_DEFAULT_SCENARIOS: list[tuple[dict, dict, dict]] = [
    # books: A values, hats: B values, balls: both medium
    ({"books": 5, "hats": 3, "balls": 2}, {"books": 4, "hats": 1, "balls": 2}, {"books": 1, "hats": 4, "balls": 2}),
    ({"books": 4, "hats": 4, "balls": 2}, {"books": 3, "hats": 1, "balls": 2}, {"books": 1, "hats": 3, "balls": 2}),
    ({"food": 6, "water": 2, "fuel": 2}, {"food": 3, "water": 1, "fuel": 3}, {"food": 1, "water": 4, "fuel": 2}),
    ({"wood": 5, "stone": 5, "gold": 1}, {"wood": 2, "stone": 1, "gold": 5}, {"wood": 1, "stone": 3, "gold": 3}),
    ({"apples": 4, "oranges": 4, "grapes": 2}, {"apples": 3, "oranges": 1, "grapes": 2}, {"apples": 1, "oranges": 3, "grapes": 2}),
]


class NegotiationTrajectoryGenerator:
    """Generates negotiation trajectories at controlled quality levels.

    Args:
        scenarios: List of (item_counts, a_values, b_values) tuples.
                   If None, uses the built-in set of 5 scenarios.
        max_rounds: Maximum negotiation rounds before auto-rejection.
        n_scenarios: Number of distinct scenarios to cycle through.
    """

    def __init__(
        self,
        scenarios: Optional[list[tuple[dict, dict, dict]]] = None,
        max_rounds: int = 10,
    ):
        self.scenarios = scenarios if scenarios is not None else _DEFAULT_SCENARIOS
        self.max_rounds = max_rounds

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(
        self, quality_level: str, n_trajectories: int, seed: int
    ) -> list[Trajectory]:
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        rng = random.Random(seed)
        trajectories = []

        for i in range(n_trajectories):
            scenario = self.scenarios[i % len(self.scenarios)]
            item_counts, a_values, b_values = scenario

            pool = ItemPool(
                counts=dict(item_counts),
                item_types=list(item_counts.keys()),
            )
            a_vals = Valuations(values=dict(a_values))
            b_vals = Valuations(values=dict(b_values))

            traj_seed = rng.randint(0, 2**31)
            traj_rng = random.Random(traj_seed)

            traj = self._generate_one(
                quality_level, pool, a_vals, b_vals, traj_rng
            )
            trajectories.append(traj)

        return trajectories

    def _generate_one(
        self,
        quality_level: str,
        pool: ItemPool,
        a_vals: Valuations,
        b_vals: Valuations,
        rng: random.Random,
    ) -> Trajectory:
        # Instantiate strategies
        strategy_cls = STRATEGY_REGISTRY[quality_level]
        if quality_level == "optimal":
            player_a = strategy_cls(opp_vals=b_vals)
        else:
            player_a = strategy_cls()

        player_b = TitForTatStrategy()

        # Game loop
        rounds: list[dict] = []
        last_a_offer: Optional[Offer] = None
        last_b_offer: Optional[Offer] = None
        final_outcome = "disagreement"
        final_deal: Optional[Offer] = None

        for r in range(self.max_rounds):
            # Player A acts
            a_action = player_a.act(r, pool, a_vals, last_b_offer, rng, self.max_rounds)

            if a_action.action_type == "accept" and last_b_offer is not None:
                final_outcome = "deal_a_accepted"
                final_deal = last_b_offer
                rounds.append({"round": r, "player": "A", "action": a_action, "offer": last_b_offer})
                break

            last_a_offer = a_action.offer
            rounds.append({"round": r, "player": "A", "action": a_action, "offer": a_action.offer})

            # Player B acts
            b_action = player_b.act(r, pool, b_vals, last_a_offer, rng, self.max_rounds)

            if b_action.action_type == "accept" and last_a_offer is not None:
                final_outcome = "deal_b_accepted"
                final_deal = last_a_offer
                rounds.append({"round": r, "player": "B", "action": b_action, "offer": last_a_offer})
                break

            last_b_offer = b_action.offer
            rounds.append({"round": r, "player": "B", "action": b_action, "offer": b_action.offer})

        # Build tokens
        tokens = self._build_tokens(rounds, pool, a_vals, b_vals)

        # Compute raw utilities for quality scorer
        if final_deal is not None:
            a_raw_util = sum(
                a_vals.values.get(item, 0) * final_deal.a_gets.get(item, 0)
                for item in pool.item_types
            )
            b_raw_util = sum(
                b_vals.values.get(item, 0) * final_deal.b_gets.get(item, 0)
                for item in pool.item_types
            )
        else:
            a_raw_util = 0.0
            b_raw_util = 0.0

        a_max = a_vals.total_value(pool)
        b_max = b_vals.total_value(pool)
        total_max = a_max + b_max

        a_util_norm = a_raw_util / a_max if a_max > 0 else 0.0
        pareto = (a_raw_util + b_raw_util) / total_max if total_max > 0 else 0.0

        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            tokens=tokens,
            character_prompt="",
            quality_scores={},
            quality_level=quality_level,
            environment="negotiation",
            metadata={
                "pool": {"counts": pool.counts, "item_types": pool.item_types},
                "a_values": a_vals.values,
                "b_values": b_vals.values,
                "rounds": [
                    {
                        "round": rec["round"],
                        "player": rec["player"],
                        "action_type": rec["action"].action_type,
                        "offer": {
                            "a_gets": rec["offer"].a_gets if rec["offer"] else None,
                            "b_gets": rec["offer"].b_gets if rec["offer"] else None,
                        } if rec.get("offer") else None,
                    }
                    for rec in rounds
                ],
                "final_outcome": final_outcome,
                "final_deal": {
                    "a_gets": final_deal.a_gets,
                    "b_gets": final_deal.b_gets,
                } if final_deal else None,
                "a_utility_normalised": a_util_norm,
                "pareto_efficiency": pareto,
                "n_rounds": len(rounds),
            },
        )

    def _build_tokens(
        self,
        rounds: list[dict],
        pool: ItemPool,
        a_vals: Valuations,
        b_vals: Valuations,
    ) -> list[Token]:
        tokens: list[Token] = []
        position = 0

        # Initial observation: scene setup
        items_desc = ", ".join(f"{v} {k}" for k, v in pool.counts.items())
        setup_text = f"Items to divide: {items_desc}."
        tokens.append(
            Token(
                text=setup_text,
                token_ids=[],
                is_observation=True,
                semantic_type="game_state",
                position=position,
            )
        )
        position += 1

        for rec in rounds:
            offer = rec.get("offer")
            action = rec["action"]

            if rec["player"] == "A":
                # Player A's action
                action_text = self._format_action(action, offer, pool)
                tokens.append(
                    Token(
                        text=action_text,
                        token_ids=[],
                        is_observation=False,
                        semantic_type="agent_proposal"
                        if action.action_type == "propose"
                        else "agent_accept",
                        position=position,
                    )
                )
                position += 1
            else:
                # Player B's response is an observation
                response_text = self._format_action(action, offer, pool)
                tokens.append(
                    Token(
                        text=response_text,
                        token_ids=[],
                        is_observation=True,
                        semantic_type="opponent_response",
                        position=position,
                    )
                )
                position += 1

        return tokens

    @staticmethod
    def _format_action(action: NegotiationAction, offer: Optional[Offer], pool: ItemPool) -> str:
        if action.action_type == "accept":
            if offer:
                a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
                b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
                return f"I accept your proposal. I get {a_parts}, you get {b_parts}."
            return "I accept your proposal."
        elif action.action_type == "propose" and offer:
            a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
            b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
            return f"I propose: I get {a_parts}, you get {b_parts}."
        elif action.action_type == "reject" and offer:
            a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
            b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
            return f"I reject your proposal. Counter-offer: I get {a_parts}, you get {b_parts}."
        return "No deal."
