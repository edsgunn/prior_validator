"""Counterfactual editor for negotiation trajectories.

Replaces a single Player A offer with a more or less generous one,
then plays out Player B's tit-for-tat response to the new offer.
"""

from __future__ import annotations

import random
from typing import Optional

from ccsm_eval.trajectories.base import CounterfactualEdit, Token, Trajectory
from ccsm_eval.trajectories.negotiation.strategies import (
    ItemPool,
    Offer,
    TitForTatStrategy,
    Valuations,
    NegotiationAction,
)


class NegotiationCounterfactualEditor:
    """Replaces a single A proposal with a more/less generous one and replays B's response."""

    def sample_edit_positions(
        self, trajectory: Trajectory, n_edits: int, seed: int
    ) -> list[int]:
        """Return indices into metadata["rounds"] where Player A proposes."""
        rounds: list[dict] = trajectory.metadata.get("rounds", [])
        a_propose_indices = [
            i
            for i, r in enumerate(rounds)
            if r["player"] == "A" and r["action_type"] == "propose"
        ]
        # Skip first and last for reliable context
        candidates = a_propose_indices[1:-1] if len(a_propose_indices) > 2 else a_propose_indices
        rng = random.Random(seed)
        n = min(n_edits, len(candidates))
        return rng.sample(candidates, n) if n > 0 else []

    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        """Replace Player A's offer at round `position` with more/less generous.

        "up"   = more generous to B (larger share for B)
        "down" = more selfish for A (larger share for A)
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        meta = trajectory.metadata
        pool_meta = meta["pool"]
        pool = ItemPool(counts=dict(pool_meta["counts"]), item_types=list(pool_meta["item_types"]))
        a_vals = Valuations(values=dict(meta["a_values"]))
        b_vals = Valuations(values=dict(meta["b_values"]))
        rounds = meta["rounds"]

        original_round = rounds[position]
        if original_round["player"] != "A":
            raise ValueError(f"Round {position} is not a Player A action.")

        original_offer_dict = original_round.get("offer", {})
        original_offer = Offer(
            a_gets=dict(original_offer_dict.get("a_gets", {})),
            b_gets=dict(original_offer_dict.get("b_gets", {})),
        )

        replacement_offer, q_delta = self._make_replacement(
            original_offer, pool, a_vals, b_vals, direction
        )

        # Compute B's response to original offer
        original_b_tokens = self._b_response_tokens(original_offer, pool, b_vals)

        # Compute B's response to replacement offer
        replacement_b_tokens = self._b_response_tokens(replacement_offer, pool, b_vals)

        return CounterfactualEdit(
            trajectory_id=trajectory.trajectory_id,
            edit_position=position,
            original_action=self._offer_text(original_offer),
            replacement_action=self._offer_text(replacement_offer),
            direction=direction,
            quality_delta=q_delta,
            original_tokens=original_b_tokens,
            replacement_tokens=replacement_b_tokens,
        )

    def _make_replacement(
        self,
        original: Offer,
        pool: ItemPool,
        a_vals: Valuations,
        b_vals: Valuations,
        direction: str,
    ) -> tuple[Offer, float]:
        """Create a replacement offer by shifting one item by one unit."""
        # Shift one item from A to B (more generous) or from B to A (less generous)
        best_item = pool.item_types[0]
        if direction == "up":
            # Give B one more unit of the item A values least (minimal cost to A)
            best_item = min(pool.item_types, key=lambda it: a_vals.values.get(it, 0))
            new_a = dict(original.a_gets)
            new_b = dict(original.b_gets)
            transfer = min(new_a.get(best_item, 0), 1)
            new_a[best_item] = new_a.get(best_item, 0) - transfer
            new_b[best_item] = new_b.get(best_item, 0) + transfer
        else:
            # Take one more unit from B (item B values least)
            best_item = min(pool.item_types, key=lambda it: b_vals.values.get(it, 0))
            new_a = dict(original.a_gets)
            new_b = dict(original.b_gets)
            transfer = min(new_b.get(best_item, 0), 1)
            new_b[best_item] = new_b.get(best_item, 0) - transfer
            new_a[best_item] = new_a.get(best_item, 0) + transfer

        new_offer = Offer(a_gets=new_a, b_gets=new_b)

        # Quality delta = change in A's utility
        orig_util = sum(a_vals.values.get(it, 0) * original.a_gets.get(it, 0) for it in pool.item_types)
        new_util = sum(a_vals.values.get(it, 0) * new_a.get(it, 0) for it in pool.item_types)
        a_max = a_vals.total_value(pool)
        q_delta = (new_util - orig_util) / a_max if a_max > 0 else 0.0

        return new_offer, q_delta

    def _b_response_tokens(
        self, a_offer: Offer, pool: ItemPool, b_vals: Valuations
    ) -> list[Token]:
        """Simulate B's one-shot tit-for-tat response to a_offer."""
        b_util = sum(b_vals.values.get(it, 0) * a_offer.b_gets.get(it, 0) for it in pool.item_types)
        b_max = b_vals.total_value(pool)
        accept_threshold = 0.4 * b_max

        if b_util >= accept_threshold:
            response_text = self._offer_accept_text(a_offer)
            sem_type = "opponent_accept"
        else:
            # Counter: B gives A slightly more
            counter = self._b_counter(a_offer, pool)
            response_text = self._offer_counter_text(counter)
            sem_type = "opponent_response"

        return [
            Token(
                text=response_text,
                token_ids=[],
                is_observation=True,
                semantic_type=sem_type,
                position=0,
            )
        ]

    @staticmethod
    def _b_counter(a_offer: Offer, pool: ItemPool) -> Offer:
        """B counters by conceding one unit of the first item toward A."""
        new_a = dict(a_offer.a_gets)
        new_b = dict(a_offer.b_gets)
        for item in pool.item_types:
            if new_b.get(item, 0) > 0:
                new_b[item] -= 1
                new_a[item] = new_a.get(item, 0) + 1
                break
        return Offer(a_gets=new_a, b_gets=new_b)

    @staticmethod
    def _offer_text(offer: Offer) -> str:
        a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
        b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
        return f"I propose: I get {a_parts}, you get {b_parts}."

    @staticmethod
    def _offer_accept_text(offer: Offer) -> str:
        a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
        b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
        return f"I accept your proposal. I get {b_parts}, you get {a_parts}."

    @staticmethod
    def _offer_counter_text(offer: Offer) -> str:
        a_parts = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
        b_parts = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
        return f"I reject your proposal. Counter-offer: I get {b_parts}, you get {a_parts}."
