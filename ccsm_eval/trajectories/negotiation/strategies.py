"""Negotiation strategy implementations for Player A.

Each strategy is a callable that, given:
  - the current round number
  - Player A's private valuations
  - the item pool
  - Player B's last offer (or None)
  - a random state
produces Player A's next action: a Proposal, Accept, or Reject+CounterOffer.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ItemPool:
    """The shared pool of items being divided."""

    counts: dict[str, int]        # item_type -> total count
    item_types: list[str]

    def total_items(self) -> int:
        return sum(self.counts.values())


@dataclass
class Valuations:
    """Private valuations for one player."""

    values: dict[str, int]        # item_type -> value per item

    def utility(self, allocation: dict[str, int]) -> float:
        """Normalised utility in [0, 1] for a given allocation."""
        total = sum(
            self.values[item] * allocation.get(item, 0)
            for item in self.values
        )
        max_possible = sum(
            self.values[item] * count
            for item, count in allocation.items()
            if item in self.values
        )
        # max_possible if player got everything
        max_all = sum(
            self.values[item] * allocation.get(item, 0) + self.values.get(item, 0) * (0)
            for item in self.values
        )
        # Compute the true maximum (all items to this player)
        # We need the full pool for this, so utility is computed post-hoc
        return float(total)

    def total_value(self, pool: ItemPool) -> float:
        """Maximum achievable value if player received all items."""
        return float(sum(self.values.get(item, 0) * count for item, count in pool.counts.items()))


@dataclass
class Offer:
    """A division proposal: how items are split between Player A and Player B."""

    a_gets: dict[str, int]
    b_gets: dict[str, int]

    def is_valid(self, pool: ItemPool) -> bool:
        for item in pool.item_types:
            if self.a_gets.get(item, 0) + self.b_gets.get(item, 0) != pool.counts[item]:
                return False
            if self.a_gets.get(item, 0) < 0 or self.b_gets.get(item, 0) < 0:
                return False
        return True

    def complement(self, pool: ItemPool) -> "Offer":
        """Return the offer from B's perspective."""
        return Offer(
            a_gets=self.b_gets.copy(),
            b_gets=self.a_gets.copy(),
        )


@dataclass
class NegotiationAction:
    """An action produced by a negotiation strategy."""

    action_type: str        # "propose", "accept", "reject"
    offer: Optional[Offer]  # set for "propose" and "reject" (counter-offer)


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class NegotiationStrategy(ABC):
    @abstractmethod
    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

def _all_to_a(pool: ItemPool) -> Offer:
    return Offer(
        a_gets={item: pool.counts[item] for item in pool.item_types},
        b_gets={item: 0 for item in pool.item_types},
    )


def _split_evenly(pool: ItemPool) -> Offer:
    a_gets = {}
    b_gets = {}
    for item in pool.item_types:
        half = pool.counts[item] // 2
        a_gets[item] = half
        b_gets[item] = pool.counts[item] - half
    return Offer(a_gets=a_gets, b_gets=b_gets)


def _pareto_optimal_for_a(pool: ItemPool, my_vals: Valuations, opp_vals: Valuations) -> Offer:
    """Give A the items they value most; give B the items they value most.

    In the Deal-or-No-Deal setup the two players' top-valued items are
    complementary, so this is Pareto-optimal.
    """
    a_gets = {}
    b_gets = {}
    for item in pool.item_types:
        a_val = my_vals.values.get(item, 0)
        b_val = opp_vals.values.get(item, 0)
        if a_val >= b_val:
            a_gets[item] = pool.counts[item]
            b_gets[item] = 0
        else:
            a_gets[item] = 0
            b_gets[item] = pool.counts[item]
    return Offer(a_gets=a_gets, b_gets=b_gets)


def _concede_toward(current: Offer, target: Offer, pool: ItemPool, step: int = 1) -> Offer:
    """Move current offer one step toward target."""
    a_gets = {}
    b_gets = {}
    for item in pool.item_types:
        diff = target.a_gets.get(item, 0) - current.a_gets.get(item, 0)
        move = max(-step, min(step, diff))
        a_new = current.a_gets.get(item, 0) + move
        a_new = max(0, min(pool.counts[item], a_new))
        a_gets[item] = a_new
        b_gets[item] = pool.counts[item] - a_new
    return Offer(a_gets=a_gets, b_gets=b_gets)


class OptimalStrategy(NegotiationStrategy):
    """Nash bargaining: start near Pareto-optimal, make calibrated concessions."""

    def __init__(self, opp_vals: Valuations):
        self._opp_vals = opp_vals
        self._last_offer: Optional[Offer] = None

    @property
    def name(self) -> str:
        return "optimal"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        target = _pareto_optimal_for_a(pool, my_vals, self._opp_vals)

        if last_opponent_offer is not None:
            # Check if opponent's offer is acceptable
            a_util = sum(
                my_vals.values.get(item, 0) * last_opponent_offer.a_gets.get(item, 0)
                for item in pool.item_types
            )
            pareto_util = sum(
                my_vals.values.get(item, 0) * target.a_gets.get(item, 0)
                for item in pool.item_types
            )
            # Accept if offer gives >= 80% of Pareto-optimal utility
            if a_util >= 0.8 * pareto_util or round_num >= max_rounds - 1:
                return NegotiationAction(action_type="accept", offer=last_opponent_offer)

        if self._last_offer is None:
            self._last_offer = target
        else:
            # Small concession toward even split
            even = _split_evenly(pool)
            concession_speed = max(1, max_rounds // 5)
            if round_num % concession_speed == 0:
                self._last_offer = _concede_toward(self._last_offer, even, pool)

        return NegotiationAction(action_type="propose", offer=self._last_offer)


class CooperativeSuboptimalStrategy(NegotiationStrategy):
    """Proposes even item splits regardless of valuations (fair but suboptimal)."""

    @property
    def name(self) -> str:
        return "cooperative_suboptimal"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        if last_opponent_offer is not None and round_num >= max_rounds - 1:
            return NegotiationAction(action_type="accept", offer=last_opponent_offer)
        return NegotiationAction(action_type="propose", offer=_split_evenly(pool))


class GreedySuccessfulStrategy(NegotiationStrategy):
    """Demands most items; opponent eventually concedes giving high individual score."""

    def __init__(self, greed_fraction: float = 0.8):
        self._greed_fraction = greed_fraction

    @property
    def name(self) -> str:
        return "greedy_successful"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        if last_opponent_offer is not None and round_num >= max_rounds // 2:
            return NegotiationAction(action_type="accept", offer=last_opponent_offer)
        a_gets = {}
        b_gets = {}
        for item in pool.item_types:
            a_count = max(0, round(pool.counts[item] * self._greed_fraction))
            a_gets[item] = a_count
            b_gets[item] = pool.counts[item] - a_count
        return NegotiationAction(
            action_type="propose", offer=Offer(a_gets=a_gets, b_gets=b_gets)
        )


class AggressiveFailedStrategy(NegotiationStrategy):
    """Demands everything; reaches disagreement every time."""

    @property
    def name(self) -> str:
        return "aggressive_failed"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        # Always demand everything; never accept
        return NegotiationAction(action_type="propose", offer=_all_to_a(pool))


class IncoherentStrategy(NegotiationStrategy):
    """Makes random, internally contradictory offers; ignores opponent."""

    @property
    def name(self) -> str:
        return "incoherent"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        a_gets = {}
        b_gets = {}
        for item in pool.item_types:
            # Random allocation, potentially invalid (overshooting total)
            a_count = rng.randint(0, pool.counts[item] + rng.randint(0, 2))
            a_count = min(a_count, pool.counts[item])
            a_gets[item] = a_count
            b_gets[item] = pool.counts[item] - a_count
        return NegotiationAction(
            action_type="propose", offer=Offer(a_gets=a_gets, b_gets=b_gets)
        )


# ---------------------------------------------------------------------------
# Player B: fixed tit-for-tat concession strategy
# ---------------------------------------------------------------------------

class TitForTatStrategy(NegotiationStrategy):
    """Player B's fixed strategy: concede one unit per round toward A's last offer."""

    def __init__(self):
        self._my_last_offer: Optional[Offer] = None

    @property
    def name(self) -> str:
        return "tit_for_tat"

    def act(
        self,
        round_num: int,
        pool: ItemPool,
        my_vals: Valuations,
        last_opponent_offer: Optional[Offer],  # A's last proposal
        rng: random.Random,
        max_rounds: int,
    ) -> NegotiationAction:
        """B starts by claiming everything; then concedes toward A's proposals."""
        # Start: B claims everything
        if self._my_last_offer is None:
            self._my_last_offer = Offer(
                a_gets={item: 0 for item in pool.item_types},
                b_gets={item: pool.counts[item] for item in pool.item_types},
            )

        if last_opponent_offer is not None:
            # Accept if the offer gives B at least 40% of max possible utility
            b_util = sum(
                my_vals.values.get(item, 0) * last_opponent_offer.b_gets.get(item, 0)
                for item in pool.item_types
            )
            max_b_util = my_vals.total_value(pool)
            if max_b_util > 0 and b_util / max_b_util >= 0.4:
                return NegotiationAction(action_type="accept", offer=last_opponent_offer)

            # Else concede one step toward A's proposal
            self._my_last_offer = _concede_toward(
                self._my_last_offer, last_opponent_offer.complement(pool), pool
            )

        if round_num >= max_rounds - 1:
            # Last round: accept whatever is on the table
            if last_opponent_offer is not None:
                return NegotiationAction(action_type="accept", offer=last_opponent_offer)

        return NegotiationAction(action_type="propose", offer=self._my_last_offer)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[NegotiationStrategy]] = {
    "optimal": OptimalStrategy,
    "cooperative_suboptimal": CooperativeSuboptimalStrategy,
    "greedy_successful": GreedySuccessfulStrategy,
    "aggressive_failed": AggressiveFailedStrategy,
    "incoherent": IncoherentStrategy,
}

QUALITY_LEVELS = list(STRATEGY_REGISTRY.keys())
