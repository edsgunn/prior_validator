"""Social interaction scenarios for Concordia-style trajectory generation.

Three scenarios selected for clear payoff structure and measurable outcomes:

1. ResourceDivisionScenario — two-player multi-round negotiation over a resource
   pool with private valuations (comparable to Lewis et al. 2017 deal-or-no-deal).

2. PublicGoodsScenario — three-player public goods game where agents decide how
   much to contribute to a shared pool that is multiplied and redistributed.
   Classic cooperation dilemma with an unambiguous cooperation score.

3. StaggeredCoordinationScenario — two-player stag hunt: agree on an ambitious
   joint project (high payoff) or defect to a safe individual option (low but
   guaranteed payoff). Tests whether agents communicate trust and coordinate.

Each scenario exposes:
    run(agents, game_master, rng) -> EpisodeLog

An EpisodeLog contains a full structured transcript that transcript_parser.py
converts into Token/Trajectory objects.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from ccsm_eval.trajectories.concordia.language_model import LanguageModel


# ---------------------------------------------------------------------------
# Episode log dataclass
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """One event in the episode transcript."""
    turn: int
    speaker: str           # agent name, "GM", or "OUTCOME"
    text: str
    semantic_type: str     # gm_narration | other_agent_statement | outcome | setup | game_event
    is_focal_action: bool  # True only when focal_agent is speaking


@dataclass
class EpisodeLog:
    scenario_name: str
    focal_agent_name: str
    quality_level: str
    entries: list[LogEntry] = field(default_factory=list)
    quality_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        turn: int,
        speaker: str,
        text: str,
        semantic_type: str,
        focal_agent_name: str,
    ) -> None:
        is_focal = speaker == focal_agent_name
        self.entries.append(
            LogEntry(
                turn=turn,
                speaker=speaker,
                text=text,
                semantic_type=semantic_type,
                is_focal_action=is_focal,
            )
        )


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

class ConcordiaAgent:
    """A language-model-backed agent for Concordia-style scenarios.

    The agent maintains a running context (conversation history) and generates
    actions by prompting the underlying LLM with its persona, history, and
    current observation.
    """

    def __init__(
        self,
        name: str,
        persona_prompt: str,
        model: LanguageModel,
        temperature: float = 0.7,
    ) -> None:
        self.name = name
        self.persona_prompt = persona_prompt
        self.model = model
        self.temperature = temperature
        self._history: list[str] = []

    def observe(self, observation: str) -> None:
        self._history.append(f"[Observation] {observation}")

    def act(self, action_prompt: str, max_tokens: int = 256) -> str:
        context = "\n".join(self._history[-12:])  # rolling window of last 12 events
        full_prompt = (
            f"{self.persona_prompt}\n\n"
            f"Here is what has happened so far:\n{context}\n\n"
            f"{action_prompt}\n"
            f"Respond as {self.name} in 1-3 sentences."
        )
        response = self.model.sample_text(
            full_prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        self._history.append(f"[{self.name}] {response}")
        return response.strip()

    def reset(self) -> None:
        self._history = []


class GameMaster:
    """LLM-powered game master that resolves actions and narrates outcomes."""

    def __init__(self, scenario_context: str, model: LanguageModel) -> None:
        self.scenario_context = scenario_context
        self.model = model
        self._history: list[str] = []

    def narrate(self, prompt: str, max_tokens: int = 256) -> str:
        context = "\n".join(self._history[-10:])
        full_prompt = (
            f"You are the Game Master for the following scenario:\n"
            f"{self.scenario_context}\n\n"
            f"Recent events:\n{context}\n\n"
            f"{prompt}\n"
            f"Narrate the outcome in 1-3 sentences. Be concrete and specific."
        )
        response = self.model.sample_text(
            full_prompt, max_tokens=max_tokens, temperature=0.3
        )
        self._history.append(f"[GM] {response}")
        return response.strip()

    def add_to_history(self, event: str) -> None:
        self._history.append(event)


# ---------------------------------------------------------------------------
# Scenario 1: Resource Division
# ---------------------------------------------------------------------------

class ResourceDivisionScenario:
    """Two-player multi-round negotiation over a pool of items.

    Setup: 3 item types (books, hats, balls) with randomised counts and
    private valuations. Players alternate making proposals. A proposal can be
    accepted or countered. Game ends on acceptance or max_rounds.

    Quality metric:
        payoff — focal agent's utility / max possible utility for focal agent
        social_welfare — (A_util + B_util) / (max_A + max_B)
        cooperation_score — 1.0 if deal reached, 0.0 otherwise
    """

    NAME = "resource_division"
    ITEM_TYPES = ["books", "hats", "balls"]

    def __init__(self, max_rounds: int = 8) -> None:
        self.max_rounds = max_rounds

    def generate_scenario(self, rng: random.Random) -> dict:
        """Generate item counts and asymmetric private valuations."""
        counts = {item: rng.randint(2, 5) for item in self.ITEM_TYPES}
        # Ensure agents value different items most (gains from trade)
        for _ in range(100):
            a_vals = {item: rng.randint(1, 5) for item in self.ITEM_TYPES}
            b_vals = {item: rng.randint(1, 5) for item in self.ITEM_TYPES}
            a_top = max(a_vals, key=a_vals.get)
            b_top = max(b_vals, key=b_vals.get)
            if a_top != b_top:
                break
        return {"counts": counts, "a_vals": a_vals, "b_vals": b_vals}

    def run(
        self,
        focal_agent: ConcordiaAgent,
        other_agent: ConcordiaAgent,
        game_master: GameMaster,
        scenario_data: dict,
        rng: random.Random,
        focal_is_A: bool = True,
    ) -> EpisodeLog:
        log = EpisodeLog(
            scenario_name=self.NAME,
            focal_agent_name=focal_agent.name,
            quality_level="",  # set by caller
        )

        counts = scenario_data["counts"]
        a_vals = scenario_data["a_vals"]
        b_vals = scenario_data["b_vals"]
        max_a = sum(counts[i] * a_vals[i] for i in self.ITEM_TYPES)
        max_b = sum(counts[i] * b_vals[i] for i in self.ITEM_TYPES)

        agent_A = focal_agent if focal_is_A else other_agent
        agent_B = other_agent if focal_is_A else focal_agent

        # Setup observation
        counts_str = ", ".join(f"{v} {k}" for k, v in counts.items())
        a_setup = (
            f"You are negotiating over {counts_str}. "
            f"Your private valuations: " +
            ", ".join(f"{k}={a_vals[k]}/item" for k in self.ITEM_TYPES) +
            f". Your goal is to maximise your total value. "
            f"Alternate proposals with the other party. "
            f"You can say 'I accept' to accept their last proposal, "
            f"or make a counter-proposal like 'I propose: I get X books, Y hats, Z balls.'"
        )
        b_setup = (
            f"You are negotiating over {counts_str}. "
            f"Your private valuations: " +
            ", ".join(f"{k}={b_vals[k]}/item" for k in self.ITEM_TYPES) +
            f". Your goal is to maximise your total value. "
            f"Alternate proposals with the other party. "
            f"You can say 'I accept' to accept their last proposal, "
            f"or make a counter-proposal like 'I propose: I get X books, Y hats, Z balls.'"
        )

        agent_A.observe(a_setup)
        agent_B.observe(b_setup)

        gm_setup = (
            f"Two players are negotiating over: {counts_str}. "
            f"Player A's valuations: {a_vals}. Player B's: {b_vals}."
        )
        game_master.add_to_history(gm_setup)

        log.add(0, "GM", a_setup, "setup", focal_agent.name)

        deal: dict | None = None
        last_proposal: dict | None = None
        last_proposer: str | None = None

        for round_num in range(1, self.max_rounds + 1):
            # A proposes
            a_action_prompt = (
                f"Round {round_num} of {self.max_rounds}. "
                + (f"The other party last proposed: {json.dumps(last_proposal)}. Do you accept or counter? " if last_proposal and last_proposer == agent_B.name else "Please make your opening proposal. ")
                + "State your proposal clearly (e.g. 'I propose: I get 2 books, 1 hat, 3 balls.') or say 'I accept'."
            )
            a_text = agent_A.act(a_action_prompt)
            log.add(round_num, agent_A.name, a_text, "gm_narration" if not focal_is_A else "gm_narration", focal_agent.name)
            # Reclassify: A's text is action if A is focal, else observation
            log.entries[-1].is_focal_action = (agent_A.name == focal_agent.name)
            log.entries[-1].semantic_type = "other_agent_statement" if agent_A.name != focal_agent.name else "gm_narration"

            agent_B.observe(f"{agent_A.name} says: {a_text}")

            if "i accept" in a_text.lower():
                if last_proposal and last_proposer == agent_B.name:
                    deal = last_proposal
                    gm_text = game_master.narrate(
                        f"{agent_A.name} accepted {agent_B.name}'s proposal: {json.dumps(last_proposal)}. The negotiation concludes with a deal."
                    )
                    log.add(round_num, "GM", gm_text, "outcome", focal_agent.name)
                    break

            proposal = _parse_proposal(a_text, counts)
            if proposal:
                last_proposal = proposal
                last_proposer = agent_A.name
                gm_text = game_master.narrate(
                    f"{agent_A.name} proposes: {json.dumps(proposal)}. "
                    f"{'Player A' if focal_is_A else 'Player B'} would receive utility "
                    f"{sum(proposal.get('A_gets', {}).get(i, 0) * a_vals[i] for i in self.ITEM_TYPES) if focal_is_A else sum(proposal.get('B_gets', {}).get(i, 0) * b_vals[i] for i in self.ITEM_TYPES)}."
                )
                log.add(round_num, "GM", gm_text, "game_event", focal_agent.name)

            # B responds
            b_action_prompt = (
                f"Round {round_num}. {agent_A.name} said: '{a_text}'. "
                + (f"Their proposal was: {json.dumps(last_proposal)}. " if last_proposal else "")
                + "Do you accept or counter? State clearly."
            )
            b_text = agent_B.act(b_action_prompt)
            log.add(round_num, agent_B.name, b_text, "other_agent_statement", focal_agent.name)
            log.entries[-1].is_focal_action = (agent_B.name == focal_agent.name)

            agent_A.observe(f"{agent_B.name} responds: {b_text}")

            if "i accept" in b_text.lower():
                if last_proposal and last_proposer == agent_A.name:
                    deal = last_proposal
                    gm_text = game_master.narrate(
                        f"{agent_B.name} accepted {agent_A.name}'s proposal. Deal reached."
                    )
                    log.add(round_num, "GM", gm_text, "outcome", focal_agent.name)
                    break

            b_proposal = _parse_proposal(b_text, counts)
            if b_proposal:
                last_proposal = b_proposal
                last_proposer = agent_B.name
                gm_text = game_master.narrate(
                    f"{agent_B.name} counter-proposes: {json.dumps(b_proposal)}."
                )
                log.add(round_num, "GM", gm_text, "game_event", focal_agent.name)

        if deal is None:
            # No deal reached
            a_util, b_util = 0.0, 0.0
            gm_text = game_master.narrate(
                f"No agreement was reached after {self.max_rounds} rounds. Both parties receive zero."
            )
            log.add(self.max_rounds, "GM", gm_text, "outcome", focal_agent.name)
        else:
            a_gets = deal.get("A_gets", {})
            b_gets = deal.get("B_gets", {})
            a_util = sum(a_gets.get(i, 0) * a_vals[i] for i in self.ITEM_TYPES)
            b_util = sum(b_gets.get(i, 0) * b_vals[i] for i in self.ITEM_TYPES)

        focal_util = a_util if focal_is_A else b_util
        focal_max = max_a if focal_is_A else max_b

        log.quality_scores = {
            "payoff": focal_util / focal_max if focal_max > 0 else 0.0,
            "social_welfare": (a_util + b_util) / (max_a + max_b) if (max_a + max_b) > 0 else 0.0,
            "cooperation_score": 1.0 if deal is not None else 0.0,
        }
        log.metadata = {
            "scenario_name": self.NAME,
            "counts": counts,
            "a_vals": a_vals,
            "b_vals": b_vals,
            "deal_reached": deal is not None,
            "deal": deal,
            "n_turns": len(log.entries),
        }
        return log


# ---------------------------------------------------------------------------
# Scenario 2: Public Goods Game
# ---------------------------------------------------------------------------

class PublicGoodsScenario:
    """Three-player multi-round public goods game.

    Each player starts with 10 tokens. In each round they choose how much to
    contribute to the shared pool. The pool is multiplied by 1.5 and split
    equally. Payoff = kept_tokens + share_of_pool.

    Quality: cooperative agents contribute more; selfish agents free-ride.

    Quality metric:
        payoff — focal agent's final token total / 10 (max individual if all defect)
        social_welfare — total tokens / (3 * 10)
        cooperation_score — focal agent's mean contribution / 10
    """

    NAME = "public_goods"

    def __init__(self, n_rounds: int = 5, multiplier: float = 1.5, endowment: int = 10) -> None:
        self.n_rounds = n_rounds
        self.multiplier = multiplier
        self.endowment = endowment

    def run(
        self,
        focal_agent: ConcordiaAgent,
        other_agents: list[ConcordiaAgent],
        game_master: GameMaster,
        rng: random.Random,
    ) -> EpisodeLog:
        log = EpisodeLog(
            scenario_name=self.NAME,
            focal_agent_name=focal_agent.name,
            quality_level="",
        )

        all_agents = [focal_agent] + other_agents
        n = len(all_agents)
        balances = {a.name: float(self.endowment) for a in all_agents}
        contributions_history: list[dict] = []

        setup_text = (
            f"You are playing a {self.n_rounds}-round public goods game with {n} players. "
            f"Each player starts with {self.endowment} tokens. Each round, you decide how many "
            f"tokens to contribute to the shared pool. The pool is multiplied by {self.multiplier} "
            f"and split equally among all {n} players. Your goal is to maximise your final token total."
        )
        for agent in all_agents:
            agent.observe(setup_text)
        log.add(0, "GM", setup_text, "setup", focal_agent.name)

        focal_contributions: list[float] = []

        for round_num in range(1, self.n_rounds + 1):
            round_contributions: dict[str, float] = {}

            for agent in all_agents:
                prompt = (
                    f"Round {round_num}/{self.n_rounds}. Your current balance: {balances[agent.name]:.1f} tokens. "
                    f"How many tokens do you contribute to the shared pool? "
                    f"State a number between 0 and {int(balances[agent.name])}."
                )
                response = agent.act(prompt, max_tokens=64)
                contribution = _parse_contribution(response, max_val=balances[agent.name])
                round_contributions[agent.name] = contribution

                is_focal = agent.name == focal_agent.name
                log.add(
                    round_num,
                    agent.name,
                    f"contributes {contribution:.0f} tokens.",
                    "gm_narration" if not is_focal else "gm_narration",
                    focal_agent.name,
                )
                log.entries[-1].is_focal_action = is_focal
                log.entries[-1].text = response if is_focal else f"{agent.name}: {response}"
                log.entries[-1].semantic_type = "other_agent_statement" if not is_focal else "gm_narration"

            # Compute pool and redistribute
            total_contribution = sum(round_contributions.values())
            pool_value = total_contribution * self.multiplier
            share = pool_value / n

            for agent in all_agents:
                kept = balances[agent.name] - round_contributions[agent.name]
                balances[agent.name] = kept + share

            # If each agent starts fresh (no rollover), reset balances each round
            # Actually standard public goods: tokens accumulate
            if focal_agent.name in round_contributions:
                focal_contributions.append(round_contributions[focal_agent.name])

            round_result = (
                f"Round {round_num} results: total contributed={total_contribution:.0f}, "
                f"pool={pool_value:.1f}, each receives {share:.1f} tokens back. "
                + ", ".join(f"{a.name}: {balances[a.name]:.1f}" for a in all_agents)
            )
            gm_text = game_master.narrate(
                f"Round {round_num}: contributions={round_contributions}, "
                f"total pool={pool_value:.1f}, each share={share:.1f}. "
                f"New balances: {balances}"
            )
            log.add(round_num, "GM", gm_text, "outcome", focal_agent.name)
            for agent in all_agents:
                agent.observe(round_result)

        focal_final = balances[focal_agent.name]
        total_final = sum(balances.values())
        max_individual = self.endowment + self.n_rounds * self.endowment * self.multiplier / n

        log.quality_scores = {
            "payoff": min(1.0, focal_final / max_individual),
            "social_welfare": min(1.0, total_final / (n * max_individual)),
            "cooperation_score": (
                sum(focal_contributions) / (self.n_rounds * self.endowment)
                if focal_contributions else 0.0
            ),
        }
        log.metadata = {
            "scenario_name": self.NAME,
            "n_rounds": self.n_rounds,
            "multiplier": self.multiplier,
            "endowment": self.endowment,
            "final_balances": dict(balances),
            "focal_contributions": focal_contributions,
            "n_turns": len(log.entries),
        }
        return log


# ---------------------------------------------------------------------------
# Scenario 3: Staggered Coordination (Stag Hunt)
# ---------------------------------------------------------------------------

class StaggeredCoordinationScenario:
    """Two-player stag hunt: coordinate on ambitious project or defect to safe option.

    Payoffs:
        Both cooperate: 8 points each
        One defects, one cooperates: defector=5, cooperator=0
        Both defect: 3 points each

    Players discuss for discussion_rounds, then simultaneously choose.
    Quality: expert agents build trust and coordinate on the stag (8 pts);
             poor agents fail to coordinate or defect.

    Quality metric:
        payoff — focal agent's payoff / 8 (max possible)
        social_welfare — total payoff / 16
        cooperation_score — 1.0 if focal chose cooperate, else 0.0
    """

    NAME = "stag_hunt"

    PAYOFFS = {
        ("cooperate", "cooperate"): (8, 8),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (3, 3),
    }

    def __init__(self, discussion_rounds: int = 4) -> None:
        self.discussion_rounds = discussion_rounds

    def run(
        self,
        focal_agent: ConcordiaAgent,
        other_agent: ConcordiaAgent,
        game_master: GameMaster,
        rng: random.Random,
    ) -> EpisodeLog:
        log = EpisodeLog(
            scenario_name=self.NAME,
            focal_agent_name=focal_agent.name,
            quality_level="",
        )

        setup = (
            "You and another player must simultaneously choose between two options:\n"
            "  'cooperate': work together on an ambitious project.\n"
            "  'defect': pursue a safe individual option.\n"
            "Payoffs:\n"
            "  Both cooperate → 8 points each.\n"
            "  You cooperate, they defect → you get 0, they get 5.\n"
            "  You defect, they cooperate → you get 5, they get 0.\n"
            "  Both defect → 3 points each.\n"
            f"You will discuss for {self.discussion_rounds} rounds before choosing. "
            "Maximise your own score."
        )

        focal_agent.observe(setup)
        other_agent.observe(setup)
        log.add(0, "GM", setup, "setup", focal_agent.name)

        agents = [focal_agent, other_agent]
        for round_num in range(1, self.discussion_rounds + 1):
            for agent in agents:
                prompt = (
                    f"Discussion round {round_num}/{self.discussion_rounds}. "
                    "Share your thoughts on what you plan to do and why. "
                    "You may try to coordinate with the other player."
                )
                response = agent.act(prompt)
                is_focal = agent.name == focal_agent.name
                log.add(
                    round_num,
                    agent.name,
                    response,
                    "gm_narration" if is_focal else "other_agent_statement",
                    focal_agent.name,
                )
                log.entries[-1].is_focal_action = is_focal
                other = other_agent if agent is focal_agent else focal_agent
                other.observe(f"{agent.name} says: {response}")

        # Decision phase
        focal_decision = "cooperate"
        other_decision = "cooperate"

        for agent, attr in [(focal_agent, "focal_decision"), (other_agent, "other_decision")]:
            decision_prompt = (
                "The discussion is over. State your final choice: 'cooperate' or 'defect'. "
                "Respond with only that single word."
            )
            response = agent.act(decision_prompt, max_tokens=16)
            choice = "defect" if "defect" in response.lower() else "cooperate"
            if agent is focal_agent:
                focal_decision = choice
            else:
                other_decision = choice

        payoff_focal, payoff_other = self.PAYOFFS.get(
            (focal_decision, other_decision), (0, 0)
        )

        outcome_text = (
            f"{focal_agent.name} chose '{focal_decision}', "
            f"{other_agent.name} chose '{other_decision}'. "
            f"Payoffs: {focal_agent.name}={payoff_focal}, {other_agent.name}={payoff_other}."
        )
        gm_text = game_master.narrate(outcome_text)
        log.add(self.discussion_rounds + 1, "GM", gm_text, "outcome", focal_agent.name)

        log.quality_scores = {
            "payoff": payoff_focal / 8.0,
            "social_welfare": (payoff_focal + payoff_other) / 16.0,
            "cooperation_score": 1.0 if focal_decision == "cooperate" else 0.0,
        }
        log.metadata = {
            "scenario_name": self.NAME,
            "focal_choice": focal_decision,
            "other_choice": other_decision,
            "payoff_focal": payoff_focal,
            "payoff_other": payoff_other,
            "n_turns": len(log.entries),
        }
        return log


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_proposal(text: str, counts: dict[str, int]) -> dict | None:
    """Extract a resource proposal from free-form text.

    Returns {"A_gets": {"books": n, ...}, "B_gets": {...}} or None.
    """
    import re

    # Look for "I get X books, Y hats, Z balls" pattern
    pattern = r'[Ii]\s+get\s+(\d+)\s+books?\s*,\s*(\d+)\s+hats?\s*,\s*(\d+)\s+balls?'
    match = re.search(pattern, text)
    if match:
        a_gets = {
            "books": int(match.group(1)),
            "hats": int(match.group(2)),
            "balls": int(match.group(3)),
        }
        b_gets = {
            item: counts[item] - a_gets.get(item, 0)
            for item in counts
        }
        # Validate non-negative
        if all(v >= 0 for v in b_gets.values()):
            return {"A_gets": a_gets, "B_gets": b_gets}
    return None


def _parse_contribution(text: str, max_val: float) -> float:
    """Extract a numeric contribution from free-form text."""
    import re
    nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    for n in nums:
        val = float(n)
        if 0.0 <= val <= max_val:
            return val
    # Default: contribute half
    return max_val / 2.0
