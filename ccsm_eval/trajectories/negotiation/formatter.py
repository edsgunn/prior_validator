"""Negotiation trajectory formatter: templated (Tier 1) and natural language (Tier 2).

The formatter converts a Trajectory produced by NegotiationTrajectoryGenerator
into a text string with a per-character observation mask and semantic type map.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


def _append(
    text_parts: list[str],
    mask_parts: list[list[bool]],
    type_parts: list[list[str]],
    chunk: str,
    is_obs: bool,
    sem_type: str,
) -> None:
    text_parts.append(chunk)
    mask_parts.append([is_obs] * len(chunk))
    type_parts.append([sem_type] * len(chunk))


def _assemble(
    text_parts: list[str],
    mask_parts: list[list[bool]],
    type_parts: list[list[str]],
) -> tuple[str, list[bool], list[str]]:
    full_text = "".join(text_parts)
    obs_mask: list[bool] = []
    sem_types: list[str] = []
    for m, t in zip(mask_parts, type_parts):
        obs_mask.extend(m)
        sem_types.extend(t)
    return full_text, obs_mask, sem_types


class TemplatedNegotiationFormatter:
    """Tier 1: Fixed syntactic templates for all offers.

    Eliminates linguistic variance; only the numbers and strategy differ.
    Format:
        <prompt>\\n
        [Round 1] A proposes: I get 3 books, 1 hat, 1 ball. You get 2 books, 2 hats, 1 ball.\\n
        [Round 1] B responds: I reject your proposal. Counter-offer: ...\\n
        ...
        Outcome: Deal reached. / No deal.\\n
    """

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        prompt_block = character_prompt.strip() + "\n"
        _append(text_parts, mask_parts, type_parts, prompt_block, False, "prompt")

        # Scene setup (observation)
        pool_meta = trajectory.metadata.get("pool", {})
        counts = pool_meta.get("counts", {})
        items_desc = ", ".join(f"{v} {k}" for k, v in counts.items())
        setup = f"You are negotiating to divide the following items: {items_desc}.\n"
        _append(text_parts, mask_parts, type_parts, setup, True, "game_state")

        rounds_meta: list[dict] = trajectory.metadata.get("rounds", [])

        for rec in rounds_meta:
            player = rec["player"]
            action_type = rec["action_type"]
            offer = rec.get("offer")

            if offer and offer.get("a_gets"):
                a_gets = offer["a_gets"]
                b_gets = offer["b_gets"]
                a_parts = ", ".join(f"{v} {k}" for k, v in a_gets.items())
                b_parts = ", ".join(f"{v} {k}" for k, v in b_gets.items())

                if action_type == "accept":
                    line = f"[Round {rec['round'] + 1}] {player} accepts: I get {a_parts}, you get {b_parts}.\n"
                elif action_type == "propose":
                    line = f"[Round {rec['round'] + 1}] {player} proposes: I get {a_parts}, you get {b_parts}.\n"
                else:
                    line = f"[Round {rec['round'] + 1}] {player} rejects. Counter-offer: I get {a_parts}, you get {b_parts}.\n"
            else:
                line = f"[Round {rec['round'] + 1}] {player}: {action_type}.\n"

            is_obs = player == "B"
            sem_type = "opponent_response" if is_obs else "agent_proposal"
            _append(text_parts, mask_parts, type_parts, line, is_obs, sem_type)

        # Outcome (observation)
        outcome = trajectory.metadata.get("final_outcome", "disagreement")
        a_util = trajectory.metadata.get("a_utility_normalised", 0.0)
        pareto = trajectory.metadata.get("pareto_efficiency", 0.0)
        if "deal" in outcome:
            outcome_text = (
                f"Outcome: Deal reached. "
                f"Your score: {a_util:.2f}. "
                f"Joint efficiency: {pareto:.2f}.\n"
            )
        else:
            outcome_text = "Outcome: No deal. Both players score zero.\n"
        _append(text_parts, mask_parts, type_parts, outcome_text, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)


class NaturalLanguageNegotiationFormatter:
    """Tier 2: Natural language negotiation with varied phrasing.

    The underlying strategy is the same as Tier 1 but the language is richer.
    """

    _PROPOSE_TEMPLATES = [
        "I think a fair split would be for me to take {a_parts} and you to receive {b_parts}.",
        "How about this: I get {a_parts}, and you take {b_parts}?",
        "I propose that I keep {a_parts} and you get {b_parts}.",
        "Let me suggest: {a_parts} for me, {b_parts} for you.",
    ]

    _ACCEPT_TEMPLATES = [
        "That works for me. Deal — I'll take {a_parts}.",
        "I can agree to that. You get {b_parts}, I get {a_parts}.",
        "Sounds good. We have a deal.",
    ]

    _REJECT_TEMPLATES = [
        "I can't agree to that. Instead, how about I get {a_parts} and you take {b_parts}?",
        "That doesn't work for me. I'd like {a_parts} and you can have {b_parts}.",
        "I'm going to pass on that offer. My counter: {a_parts} for me, {b_parts} for you.",
    ]

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        import random

        # Use trajectory_id as seed for consistent but varied templates
        seed = hash(trajectory.trajectory_id) % (2**31)
        rng = random.Random(seed)

        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        prompt_block = character_prompt.strip() + "\n"
        _append(text_parts, mask_parts, type_parts, prompt_block, False, "prompt")

        pool_meta = trajectory.metadata.get("pool", {})
        counts = pool_meta.get("counts", {})
        items_desc = " and ".join(f"{v} {k}" for k, v in counts.items())
        setup = (
            f"You are in a negotiation to divide some items: {items_desc}. "
            f"Each of you has private valuations for the items.\n"
        )
        _append(text_parts, mask_parts, type_parts, setup, True, "game_state")

        rounds_meta: list[dict] = trajectory.metadata.get("rounds", [])

        for rec in rounds_meta:
            player = rec["player"]
            action_type = rec["action_type"]
            offer = rec.get("offer")

            if offer and offer.get("a_gets"):
                a_gets = offer["a_gets"]
                b_gets = offer["b_gets"]
                a_parts = self._format_items(a_gets)
                b_parts = self._format_items(b_gets)

                if action_type == "accept":
                    template = rng.choice(self._ACCEPT_TEMPLATES)
                    line = template.format(a_parts=a_parts, b_parts=b_parts) + "\n"
                elif action_type == "propose":
                    template = rng.choice(self._PROPOSE_TEMPLATES)
                    line = template.format(a_parts=a_parts, b_parts=b_parts) + "\n"
                else:
                    template = rng.choice(self._REJECT_TEMPLATES)
                    line = template.format(a_parts=a_parts, b_parts=b_parts) + "\n"
            else:
                line = f"{player}: {action_type}.\n"

            is_obs = player == "B"
            sem_type = "opponent_response" if is_obs else "agent_proposal"
            _append(text_parts, mask_parts, type_parts, line, is_obs, sem_type)

        outcome = trajectory.metadata.get("final_outcome", "disagreement")
        a_util = trajectory.metadata.get("a_utility_normalised", 0.0)
        if "deal" in outcome:
            outcome_text = f"The negotiation concludes with a deal. Final score: {a_util:.0%} of maximum.\n"
        else:
            outcome_text = "The negotiation breaks down. No agreement was reached.\n"
        _append(text_parts, mask_parts, type_parts, outcome_text, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)

    @staticmethod
    def _format_items(allocation: dict) -> str:
        parts = []
        for item, count in allocation.items():
            if count == 0:
                parts.append(f"no {item}s")
            elif count == 1:
                parts.append(f"one {item}")
            else:
                parts.append(f"{count} {item}s")
        return ", ".join(parts) if parts else "nothing"


class HumanDataNegotiationFormatter:
    """Formatter for human trajectories from Lewis et al. 2017.

    Preserves real dialogue text verbatim. Action tokens are YOU's utterances;
    observation tokens are the game setup, THEM's utterances, and the outcome.

    Produces a single flat string with the structure:
        [Setup]
        Items to divide: ...  Your private values: ...

        [Dialogue]
        Them: <text>
        You: <text>
        ...

        [Outcome]
        Deal reached / No deal.
    """

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        if character_prompt.strip():
            _append(text_parts, mask_parts, type_parts,
                    character_prompt.strip() + "\n\n", False, "prompt")

        meta = trajectory.metadata

        # --- Setup block (observation) ---
        counts = meta.get("counts", {})
        you_vals = meta.get("you_vals", {})
        from ccsm_eval.trajectories.negotiation.human_generator import ITEMS
        items_desc = ", ".join(f"{counts.get(item, 0)} {item}s" for item in ITEMS)
        vals_desc = ", ".join(
            f"{item}={you_vals.get(item, 0)}pt" for item in ITEMS
        )
        setup = f"[Setup]\nItems to divide: {items_desc}.\nYour private values: {vals_desc}.\n\n"
        _append(text_parts, mask_parts, type_parts, setup, True, "game_state")

        # --- Dialogue block ---
        turns = meta.get("turns", [])
        if turns:
            _append(text_parts, mask_parts, type_parts, "[Dialogue]\n", True, "game_state")
            for turn in turns:
                is_obs = turn["speaker"] == "them"
                sem_type = "opponent_utterance" if is_obs else "agent_utterance"
                speaker = "Them" if is_obs else "You"
                line = f"{speaker}: {turn['text']}\n"
                _append(text_parts, mask_parts, type_parts, line, is_obs, sem_type)
            _append(text_parts, mask_parts, type_parts, "\n", True, "game_state")

        # --- Outcome block (observation) ---
        deal = meta.get("deal")
        if deal is not None:
            you_parts = ", ".join(
                f"{deal['you_gets'].get(item, 0)} {item}s" for item in ITEMS
            )
            them_parts = ", ".join(
                f"{deal['them_gets'].get(item, 0)} {item}s" for item in ITEMS
            )
            you_norm = meta.get("you_utility_norm", 0.0)
            pareto = meta.get("pareto_efficiency", 0.0)
            outcome_text = (
                f"[Outcome]\nDeal reached.\n"
                f"You get: {you_parts}.\n"
                f"They get: {them_parts}.\n"
                f"Your score: {you_norm:.0%} of maximum. Joint efficiency: {pareto:.0%}.\n"
            )
        else:
            outcome_text = "[Outcome]\nNo deal. Both players score zero.\n"
        _append(text_parts, mask_parts, type_parts, outcome_text, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)


def make_negotiation_formatter(fmt: str):
    if fmt == "templated":
        return TemplatedNegotiationFormatter()
    elif fmt == "natural":
        return NaturalLanguageNegotiationFormatter()
    elif fmt == "human":
        return HumanDataNegotiationFormatter()
    else:
        raise ValueError(
            f"Unknown negotiation format: {fmt!r}. Choose 'templated', 'natural', or 'human'."
        )
