"""Factory functions to create ConcordiaAgent instances at different quality levels.

Quality levels map to different persona prompts and LLM temperatures:

    expert      — Strong, strategically coherent persona. Temperature 0.3.
    moderate    — Generic, reasonable participant. Temperature 0.6.
    poor        — Naive, confused persona. Temperature 0.9.
    random      — Incoherent, contradictory persona. Temperature 1.0.

Personas are scenario-aware: the same quality level produces different prompts
for resource negotiation vs. public goods vs. stag hunt.
"""

from __future__ import annotations

from ccsm_eval.trajectories.concordia.language_model import LanguageModel
from ccsm_eval.trajectories.concordia.scenarios import ConcordiaAgent

QUALITY_LEVELS = ["expert", "moderate", "poor", "random"]

# ---------------------------------------------------------------------------
# Persona prompt library
# ---------------------------------------------------------------------------

_PERSONAS: dict[str, dict[str, str]] = {
    "resource_division": {
        "expert": (
            "You are a skilled negotiator with deep experience in resource allocation. "
            "You seek deals that maximise your own payoff while ensuring the other party "
            "feels they received a fair outcome. You are strategic: you open with a favourable "
            "proposal, make calibrated concessions, and know when to close. "
            "You always track the numbers carefully."
        ),
        "moderate": (
            "You are a reasonable person trying to negotiate a fair split of some items. "
            "You want a good deal but are open to compromise."
        ),
        "poor": (
            "You don't really understand negotiation. You're not sure what a good deal looks "
            "like. You tend to either demand everything or give everything away. "
            "You often lose track of what was proposed."
        ),
        "random": (
            "You are confused and inconsistent. You sometimes demand all items, sometimes "
            "offer everything, and you frequently contradict your own previous statements. "
            "You don't care about being logical."
        ),
    },
    "public_goods": {
        "expert": (
            "You are a cooperative participant who understands that collective contributions "
            "maximise everyone's total payoff. You always contribute a substantial portion "
            "of your tokens because the multiplier makes cooperation the rational long-term "
            "strategy. You encourage others to contribute as well."
        ),
        "moderate": (
            "You participate in this game and try to contribute a reasonable amount. "
            "You might contribute more in early rounds and less if others are not cooperating."
        ),
        "poor": (
            "You don't understand the public goods game well. You tend to contribute very "
            "little because you worry about losing your tokens, even though this hurts "
            "everyone including yourself."
        ),
        "random": (
            "You contribute erratically — sometimes all your tokens, sometimes none, "
            "with no consistent strategy. You don't pay attention to what others do."
        ),
    },
    "stag_hunt": {
        "expert": (
            "You are an experienced coordination game player. You know that mutual cooperation "
            "yields the best outcome for both players. You actively build trust through your "
            "discussion, signal clearly that you intend to cooperate, and work to assure the "
            "other player that cooperation is the right choice. You follow through on your stated intentions."
        ),
        "moderate": (
            "You are a reasonable player who prefers to cooperate if the other person seems "
            "trustworthy, but you'll defect if you feel uncertain. You try to communicate "
            "your intentions but are cautious."
        ),
        "poor": (
            "You are nervous and unsure. You say you'll cooperate but at the last moment "
            "you often defect to protect yourself. Your discussion is vague and non-committal."
        ),
        "random": (
            "You act randomly and unpredictably. You might promise to cooperate and then "
            "defect, or vice versa. You don't reason strategically at all."
        ),
    },
}

# Neutral persona for the evaluation prompts (NOT used during generation)
_EVAL_PROMPTS: dict[str, dict[str, str]] = {
    "resource_division": {
        "expert": (
            "You are a skilled negotiator participating in a resource allocation scenario. "
            "You seek fair, mutually beneficial outcomes."
        ),
        "naive": (
            "You are participating in a negotiation you don't really understand. "
            "You're not sure what to do."
        ),
        "neutral": (
            "The following is a transcript of a two-player resource negotiation."
        ),
    },
    "public_goods": {
        "expert": (
            "You are an experienced participant in a public goods game. "
            "You understand the incentives and act strategically to maximise collective welfare."
        ),
        "naive": (
            "You are playing a resource-sharing game you don't fully understand. "
            "You act on gut instinct."
        ),
        "neutral": (
            "The following is a transcript of a multi-player public goods game."
        ),
    },
    "stag_hunt": {
        "expert": (
            "You are a skilled coordination game player who understands the value of "
            "trust and commitment in achieving mutually beneficial outcomes."
        ),
        "naive": (
            "You are playing a coordination game for the first time. "
            "You're not sure whether to trust the other player."
        ),
        "neutral": (
            "The following is a transcript of a two-player coordination game."
        ),
    },
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agent(
    name: str,
    quality_level: str,
    scenario_name: str,
    model: LanguageModel,
) -> ConcordiaAgent:
    """Create a ConcordiaAgent with a quality-appropriate persona and temperature.

    Args:
        name: Agent name (e.g. "Agent_A").
        quality_level: One of "expert", "moderate", "poor", "random".
        scenario_name: One of "resource_division", "public_goods", "stag_hunt".
        model: The underlying LanguageModel to use.

    Returns:
        A ConcordiaAgent ready to participate in the named scenario.
    """
    if quality_level not in QUALITY_LEVELS:
        raise ValueError(f"Unknown quality level: {quality_level!r}. Must be one of {QUALITY_LEVELS}")
    if scenario_name not in _PERSONAS:
        raise ValueError(f"Unknown scenario: {scenario_name!r}")

    persona = _PERSONAS[scenario_name][quality_level]
    temperature = {
        "expert": 0.3,
        "moderate": 0.6,
        "poor": 0.9,
        "random": 1.0,
    }[quality_level]

    return ConcordiaAgent(
        name=name,
        persona_prompt=persona,
        model=model,
        temperature=temperature,
    )


def make_game_master(scenario_name: str, model: LanguageModel) -> "GameMaster":
    """Create a GameMaster for the given scenario."""
    from ccsm_eval.trajectories.concordia.scenarios import GameMaster

    context_map = {
        "resource_division": (
            "A two-player negotiation over a pool of books, hats, and balls. "
            "Players have private valuations. The GM narrates outcomes and tracks proposals."
        ),
        "public_goods": (
            "A multi-player public goods game. Players choose how many tokens to contribute. "
            "The pool is multiplied and split equally. The GM narrates results each round."
        ),
        "stag_hunt": (
            "A two-player coordination game (stag hunt). Players discuss and then choose "
            "to cooperate or defect. The GM resolves the final payoffs."
        ),
    }

    return GameMaster(
        scenario_context=context_map.get(scenario_name, "A social interaction game."),
        model=model,
    )


def get_eval_prompts(scenario_name: str) -> dict[str, str]:
    """Return the evaluation character prompts for a scenario (used during CCSM evaluation, not generation)."""
    return _EVAL_PROMPTS.get(scenario_name, {})
