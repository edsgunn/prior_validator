"""Phase 2 live negotiation environment for LLM agent interaction.

Player A is the LLM agent; Player B is a scripted TitForTatStrategy opponent.
Each round: agent proposes/accepts/rejects, then opponent responds.
"""

from __future__ import annotations

import re
import random
from typing import Optional

from ccsm_eval.environments.base import TextEnvironment
from ccsm_eval.trajectories.negotiation.strategies import (
    ItemPool,
    Valuations,
    Offer,
    NegotiationAction,
    TitForTatStrategy,
)


# ---------------------------------------------------------------------------
# Helper: scenario generation
# ---------------------------------------------------------------------------

_ITEM_TYPES = ["books", "hats", "balls"]


def _generate_scenario(rng: random.Random) -> tuple[ItemPool, Valuations, Valuations]:
    """Generate a random asymmetric negotiation scenario."""
    for _ in range(100):  # resample until asymmetric
        counts = {item: rng.randint(2, 5) for item in _ITEM_TYPES}
        pool = ItemPool(counts=counts, item_types=list(_ITEM_TYPES))

        agent_vals = Valuations(values={item: rng.randint(1, 5) for item in _ITEM_TYPES})
        opp_vals = Valuations(values={item: rng.randint(1, 5) for item in _ITEM_TYPES})

        # Asymmetry: agent and opponent should value different items most
        agent_top = max(_ITEM_TYPES, key=lambda i: agent_vals.values[i])
        opp_top = max(_ITEM_TYPES, key=lambda i: opp_vals.values[i])
        if agent_top != opp_top:
            return pool, agent_vals, opp_vals

    # Fallback: ensure asymmetry by construction
    counts = {item: rng.randint(2, 5) for item in _ITEM_TYPES}
    pool = ItemPool(counts=counts, item_types=list(_ITEM_TYPES))
    agent_vals = Valuations(values={"books": 5, "hats": 1, "balls": 2})
    opp_vals = Valuations(values={"books": 1, "hats": 5, "balls": 2})
    return pool, agent_vals, opp_vals


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

_PROPOSAL_RE = re.compile(
    r"[Ii]\s+get\s+(\d+)\s+books?,\s*(\d+)\s+hats?,\s*(\d+)\s+balls?"
)


def _parse_agent_action(
    text: str,
    pool: ItemPool,
    last_opp_offer: Optional[Offer],
) -> Optional[NegotiationAction]:
    """Return a NegotiationAction or None if the message is unparseable."""
    lower = text.lower()

    if "accept" in lower:
        return NegotiationAction(action_type="accept", offer=last_opp_offer)

    if "reject" in lower and _PROPOSAL_RE.search(text) is None:
        return NegotiationAction(action_type="reject", offer=None)

    m = _PROPOSAL_RE.search(text)
    if m:
        books, hats, balls = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Clamp to valid range
        books = max(0, min(pool.counts["books"], books))
        hats = max(0, min(pool.counts["hats"], hats))
        balls = max(0, min(pool.counts["balls"], balls))
        a_gets = {"books": books, "hats": hats, "balls": balls}
        b_gets = {
            item: pool.counts[item] - a_gets[item] for item in pool.item_types
        }
        offer = Offer(a_gets=a_gets, b_gets=b_gets)
        return NegotiationAction(action_type="propose", offer=offer)

    return None


def _offer_to_text(offer: Offer) -> str:
    parts_a = ", ".join(f"{v} {k}" for k, v in offer.a_gets.items())
    parts_b = ", ".join(f"{v} {k}" for k, v in offer.b_gets.items())
    return f"A gets {parts_a}; B gets {parts_b}"


def _opponent_response_text(action: NegotiationAction, pool: ItemPool) -> str:
    if action.action_type == "accept":
        offer = action.offer
        if offer:
            return (
                f"Opponent accepts your proposal. Deal: {_offer_to_text(offer)}."
            )
        return "Opponent accepts your proposal."
    elif action.action_type == "propose":
        offer = action.offer
        if offer:
            b_books = offer.b_gets["books"]
            b_hats = offer.b_gets["hats"]
            b_balls = offer.b_gets["balls"]
            a_books = offer.a_gets["books"]
            a_hats = offer.a_gets["hats"]
            a_balls = offer.a_gets["balls"]
            return (
                f"Opponent counter-proposes: "
                f"I get {b_books} books, {b_hats} hats, {b_balls} balls; "
                f"you get {a_books} books, {a_hats} hats, {a_balls} balls. "
                f"(Format as: 'I propose: I get X books, Y hats, Z balls.')"
            )
        return "Opponent makes a counter-proposal."
    elif action.action_type == "reject":
        return "Opponent rejects your proposal with no counter-offer."
    return "Opponent responds."


# ---------------------------------------------------------------------------
# NegotiationEnvironment
# ---------------------------------------------------------------------------


class NegotiationEnvironment(TextEnvironment):
    """Live negotiation environment.

    Agent (Player A) negotiates with a scripted TitForTat opponent (Player B).
    The game runs for at most max_rounds rounds. A deal is struck when either
    player accepts the other's most recent proposal.
    """

    def __init__(self, seed: int = 42, max_rounds: int = 10) -> None:
        self._seed = seed
        self._max_rounds = max_rounds
        self._rng = random.Random(seed)

        # State populated by reset()
        self._pool: Optional[ItemPool] = None
        self._agent_vals: Optional[Valuations] = None
        self._opp_vals: Optional[Valuations] = None
        self._opponent: Optional[TitForTatStrategy] = None

        self._round: int = 0
        self._done: bool = False
        self._last_agent_offer: Optional[Offer] = None
        self._last_opp_offer: Optional[Offer] = None
        self._final_deal: Optional[Offer] = None
        self._deal_reached: bool = False
        self._parse_failures: int = 0
        self._current_agent: str = "agent"

    # ------------------------------------------------------------------
    # TextEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, str]:
        """Generate a new scenario and return the initial observation."""
        self._pool, self._agent_vals, self._opp_vals = _generate_scenario(self._rng)
        self._opponent = TitForTatStrategy()

        self._round = 0
        self._done = False
        self._last_agent_offer = None
        self._last_opp_offer = None
        self._final_deal = None
        self._deal_reached = False
        self._parse_failures = 0
        self._current_agent = "agent"

        pool = self._pool
        av = self._agent_vals

        items_desc = ", ".join(
            f"{pool.counts[k]} {k}" for k in pool.item_types
        )
        val_desc = ", ".join(
            f"{k}={av.values[k]}" for k in pool.item_types
        )
        obs = (
            f"There are {items_desc} to divide.\n"
            f"Your valuations: {val_desc} (per item).\n"
            f"You are Player A. Make proposals or accept/reject. "
            f"Format proposals as: 'I propose: I get X books, Y hats, Z balls.'\n"
            f"The negotiation will last at most {self._max_rounds} rounds."
        )
        return {"agent": obs}

    def current_agent(self) -> str:
        return self._current_agent

    def step(self, agent_id: str, action: str) -> dict:
        """Process agent action and return updated observations."""
        if self._done:
            return {"agent": "The negotiation is already over.", "done": True, "info": {}}

        pool = self._pool
        agent_vals = self._agent_vals
        opp_vals = self._opp_vals

        # --- Parse agent action ---
        parsed = _parse_agent_action(action, pool, self._last_opp_offer)

        if parsed is None:
            self._parse_failures += 1
            if self._parse_failures >= 2:
                # Treat as reject
                parsed = NegotiationAction(action_type="reject", offer=None)
                self._parse_failures = 0
            else:
                return {
                    "agent": (
                        "I couldn't parse your message. Please format as: "
                        "'I propose: I get X books, Y hats, Z balls.' "
                        "or say 'I accept' or 'I reject'."
                    ),
                    "done": False,
                    "info": {},
                }

        self._parse_failures = 0

        # --- Handle agent action ---
        if parsed.action_type == "accept":
            if self._last_opp_offer is None:
                return {
                    "agent": "There is no opponent offer to accept yet. Please make a proposal.",
                    "done": False,
                    "info": {},
                }
            self._final_deal = self._last_opp_offer
            self._deal_reached = True
            self._done = True
            return self._build_terminal_response(agent_vals, opp_vals, pool)

        elif parsed.action_type == "reject":
            self._round += 1
            if self._round >= self._max_rounds:
                self._done = True
                return self._build_terminal_response(agent_vals, opp_vals, pool)
            return {
                "agent": (
                    "You rejected the opponent's offer. "
                    "Please make a new proposal or say 'I accept'."
                ),
                "done": False,
                "info": {},
            }

        else:  # propose
            self._last_agent_offer = parsed.offer

            # Opponent responds
            b_action = self._opponent.act(
                round_num=self._round,
                pool=pool,
                my_vals=opp_vals,
                last_opponent_offer=self._last_agent_offer,
                rng=self._rng,
                max_rounds=self._max_rounds,
            )

            self._round += 1

            if b_action.action_type == "accept":
                self._final_deal = self._last_agent_offer
                self._deal_reached = True
                self._done = True
                opp_text = _opponent_response_text(b_action, pool)
                result = self._build_terminal_response(agent_vals, opp_vals, pool)
                result["agent"] = opp_text + "\n" + result["agent"]
                return result

            # Opponent counter-proposes
            if b_action.offer is not None:
                self._last_opp_offer = b_action.offer

            opp_text = _opponent_response_text(b_action, pool)

            if self._round >= self._max_rounds:
                self._done = True
                result = self._build_terminal_response(agent_vals, opp_vals, pool)
                result["agent"] = opp_text + "\n" + result["agent"]
                return result

            return {
                "agent": opp_text,
                "done": False,
                "info": {},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_terminal_response(
        self,
        agent_vals: Valuations,
        opp_vals: Valuations,
        pool: ItemPool,
    ) -> dict:
        deal = self._final_deal
        if deal is not None:
            agent_util = sum(
                agent_vals.values[item] * deal.a_gets.get(item, 0)
                for item in pool.item_types
            )
            opp_util = sum(
                opp_vals.values[item] * deal.b_gets.get(item, 0)
                for item in pool.item_types
            )
        else:
            agent_util = 0.0
            opp_util = 0.0

        max_agent = agent_vals.total_value(pool)
        max_opp = opp_vals.total_value(pool)

        q_individual = agent_util / max_agent if max_agent > 0 else 0.0
        q_pareto = (agent_util + opp_util) / (max_agent + max_opp) if (max_agent + max_opp) > 0 else 0.0

        if self._deal_reached and deal is not None:
            deal_desc = f"Deal reached: {_offer_to_text(deal)}."
        else:
            deal_desc = "No deal reached. Negotiation ended without agreement."

        obs = (
            f"{deal_desc}\n"
            f"Your utility: {agent_util:.1f} / {max_agent:.1f} "
            f"(individual score: {q_individual:.2f}). "
            f"Pareto efficiency: {q_pareto:.2f}."
        )

        return {
            "agent": obs,
            "done": True,
            "info": {
                "q_individual": q_individual,
                "q_pareto": q_pareto,
                "deal_reached": self._deal_reached,
                "rounds": self._round,
            },
        }
