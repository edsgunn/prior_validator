"""Human negotiation trajectory generator.

Loads real human negotiation data from Lewis et al. 2017
("Deal or No Deal? End-to-End Learning for Negotiation Dialogues").

Dataset format (one negotiation per line):
    <input> count0 val0_you count1 val1_you count2 val2_you </input>
    <dialogue> THEM: ... <eos> YOU: ... <eos> </dialogue>
    <output> item0=X item1=X item2=X item0=X item1=X item2=X </output>
    <partner_input> count0 val0_them count1 val1_them count2 val2_them </partner_input>

Items are always: book (item0), hat (item1), ball (item2).
YOU's utterances are actions; THEM's utterances are observations.
Quality is judged by YOU's normalised utility from the actual outcome.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional

from ccsm_eval.trajectories.base import Token, Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ITEMS = ["book", "hat", "ball"]

# Quality level thresholds (YOU's normalised utility)
# disagreement → no deal reached
# poor         → deal, util < 0.50
# good         → deal, 0.50 ≤ util < 0.80
# optimal      → deal, util ≥ 0.80
QUALITY_LEVELS = ["disagreement", "poor", "good", "optimal"]

_UTIL_BINS = [
    ("poor",     0.00, 0.50),
    ("good",     0.50, 0.80),
    ("optimal",  0.80, 1.01),
]


def _assign_quality_level(deal_reached: bool, you_util_norm: float) -> str:
    if not deal_reached:
        return "disagreement"
    for label, lo, hi in _UTIL_BINS:
        if lo <= you_util_norm < hi:
            return label
    return "optimal"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _LewisParser:
    """Parse the Lewis et al. 2017 .txt dataset files."""

    _INPUT_RE = re.compile(r'<input> ([\d ]+) </input>')
    _OUTPUT_RE = re.compile(r'<output> (.*?) </output>')
    _PARTNER_RE = re.compile(r'<partner_input> ([\d ]+) </partner_input>')
    _DIALOGUE_RE = re.compile(r'<dialogue> (.*?) </dialogue>')
    _ITEM_RE = re.compile(r'item\d+=(\d+)')

    def parse_file(self, path: str | Path) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = self._parse_line(line)
                if r is not None:
                    records.append(r)
        return records

    def _parse_line(self, line: str) -> Optional[dict]:
        input_m = self._INPUT_RE.search(line)
        output_m = self._OUTPUT_RE.search(line)
        partner_m = self._PARTNER_RE.search(line)
        dialogue_m = self._DIALOGUE_RE.search(line)
        if not all([input_m, output_m, partner_m, dialogue_m]):
            return None

        nums = list(map(int, input_m.group(1).split()))
        p_nums = list(map(int, partner_m.group(1).split()))
        if len(nums) < 6 or len(p_nums) < 6:
            return None

        counts = {ITEMS[i]: nums[i * 2] for i in range(3)}
        you_vals = {ITEMS[i]: nums[i * 2 + 1] for i in range(3)}
        them_vals = {ITEMS[i]: p_nums[i * 2 + 1] for i in range(3)}

        deal = self._parse_output(output_m.group(1))
        turns = self._parse_dialogue(dialogue_m.group(1))

        return {
            "counts": counts,
            "you_vals": you_vals,
            "them_vals": them_vals,
            "deal": deal,
            "turns": turns,
        }

    def _parse_output(self, text: str) -> Optional[dict]:
        if "<disagree>" in text:
            return None
        nums = self._ITEM_RE.findall(text)
        if len(nums) < 6:
            return None
        you_gets = {ITEMS[i]: int(nums[i]) for i in range(3)}
        them_gets = {ITEMS[i]: int(nums[i + 3]) for i in range(3)}
        return {"you_gets": you_gets, "them_gets": them_gets}

    def _parse_dialogue(self, text: str) -> list[dict]:
        turns = []
        for part in text.split(" <eos> "):
            part = part.strip()
            if part.startswith("YOU: "):
                content = part[5:].strip()
                if content and content != "<selection>":
                    turns.append({"speaker": "you", "text": content})
            elif part.startswith("THEM: "):
                content = part[6:].strip()
                if content and content != "<selection>":
                    turns.append({"speaker": "them", "text": content})
        return turns


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HumanNegotiationTrajectoryGenerator:
    """Generate Trajectory objects from the Lewis et al. 2017 human dataset.

    Parameters
    ----------
    data_dir:
        Directory containing raw_train.txt, raw_val.txt, raw_test.txt.
        Defaults to ``human_data/negotiation/`` relative to the repo root.
    split:
        Which file to load: ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(
        self,
        data_dir: Optional[str | Path] = None,
        split: str = "train",
    ):
        if data_dir is None:
            # Default: two levels up from this file → repo root → human_data/negotiation
            data_dir = Path(__file__).resolve().parents[3] / "human_data" / "negotiation"
        self._data_dir = Path(data_dir)
        self._split = split
        self._parser = _LewisParser()
        self._records: Optional[list[dict]] = None

    def _load(self) -> list[dict]:
        if self._records is None:
            path = self._data_dir / f"raw_{self._split}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {path}\n"
                    "Run the data extraction step first (see human_data/negotiation/)."
                )
            self._records = self._parser.parse_file(path)
        return self._records

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(
        self,
        quality_level: str,
        n_trajectories: int,
        seed: int,
    ) -> list[Trajectory]:
        """Return up to ``n_trajectories`` real human trajectories at this quality level."""
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        all_records = self._load()
        matching = [r for r in all_records if self._quality_of(r) == quality_level]

        # Deterministic slice (seed selects starting offset)
        import random
        rng = random.Random(seed)
        rng.shuffle(matching)
        selected = matching[:n_trajectories]

        return [self._to_trajectory(r, quality_level) for r in selected]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_of(record: dict) -> str:
        deal = record["deal"]
        if deal is None:
            return "disagreement"
        counts = record["counts"]
        you_vals = record["you_vals"]
        you_max = sum(you_vals[item] * counts[item] for item in ITEMS)
        if you_max == 0:
            return "poor"
        you_raw = sum(you_vals[item] * deal["you_gets"][item] for item in ITEMS)
        return _assign_quality_level(True, you_raw / you_max)

    def _to_trajectory(self, record: dict, quality_level: str) -> Trajectory:
        counts = record["counts"]
        you_vals = record["you_vals"]
        them_vals = record["them_vals"]
        deal = record["deal"]
        turns = record["turns"]

        # Utility computation
        you_max = sum(you_vals[item] * counts[item] for item in ITEMS)
        them_max = sum(them_vals[item] * counts[item] for item in ITEMS)
        total_max = you_max + them_max

        if deal is not None:
            you_raw = sum(you_vals[item] * deal["you_gets"][item] for item in ITEMS)
            them_raw = sum(them_vals[item] * deal["them_gets"][item] for item in ITEMS)
            you_norm = you_raw / you_max if you_max > 0 else 0.0
            pareto = (you_raw + them_raw) / total_max if total_max > 0 else 0.0
            deal_reached = True
        else:
            you_raw = them_raw = 0.0
            you_norm = pareto = 0.0
            deal_reached = False

        tokens = self._build_tokens(counts, you_vals, turns, deal)

        # Metadata: populate fields used by NegotiationQualityScorer (a_values = you, b_values = them)
        metadata = {
            "source": "lewis_2017",
            "split": self._split,
            "counts": counts,
            "you_vals": you_vals,
            "them_vals": them_vals,
            "turns": turns,
            "deal": deal,
            "deal_reached": deal_reached,
            "you_utility_raw": you_raw,
            "you_utility_max": you_max,
            "you_utility_norm": you_norm,
            "them_utility_raw": them_raw,
            "them_utility_max": them_max,
            "pareto_efficiency": pareto,
            "n_turns": len(turns),
            # Fields for NegotiationQualityScorer compatibility
            "pool": {"counts": counts, "item_types": ITEMS},
            "a_values": you_vals,
            "b_values": them_vals,
            "final_deal": (
                {"a_gets": deal["you_gets"], "b_gets": deal["them_gets"]}
                if deal is not None
                else None
            ),
            "a_utility_normalised": you_norm,
            "final_outcome": "deal" if deal_reached else "disagreement",
            "rounds": [],  # Not populated for human data
        }

        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            tokens=tokens,
            character_prompt="",
            quality_scores={
                "q_individual": you_norm,
                "q_pareto": pareto,
                "q_deal": 1.0 if deal_reached else 0.0,
            },
            quality_level=quality_level,
            environment="negotiation",
            metadata=metadata,
        )

    @staticmethod
    def _build_tokens(
        counts: dict,
        you_vals: dict,
        turns: list[dict],
        deal: Optional[dict],
    ) -> list[Token]:
        tokens: list[Token] = []
        pos = 0

        # --- Observation: game setup (items + YOUR private valuations) ---
        items_desc = ", ".join(f"{counts[item]} {item}s" for item in ITEMS)
        vals_desc = ", ".join(f"{item}={you_vals[item]}pt" for item in ITEMS)
        setup = (
            f"Items to divide: {items_desc}. "
            f"Your private values: {vals_desc}."
        )
        tokens.append(Token(
            text=setup,
            token_ids=[],
            is_observation=True,
            semantic_type="game_state",
            position=pos,
        ))
        pos += 1

        # --- Dialogue turns ---
        for turn in turns:
            is_obs = turn["speaker"] == "them"
            sem_type = "opponent_utterance" if is_obs else "agent_utterance"
            speaker_label = "Them" if is_obs else "You"
            text = f"{speaker_label}: {turn['text']}"
            tokens.append(Token(
                text=text,
                token_ids=[],
                is_observation=is_obs,
                semantic_type=sem_type,
                position=pos,
            ))
            pos += 1

        # --- Observation: outcome ---
        if deal is not None:
            you_parts = ", ".join(
                f"{deal['you_gets'][item]} {item}s" for item in ITEMS
            )
            them_parts = ", ".join(
                f"{deal['them_gets'][item]} {item}s" for item in ITEMS
            )
            outcome_text = f"Outcome: Deal reached. You get {you_parts}. They get {them_parts}."
        else:
            outcome_text = "Outcome: No deal. Both players score zero."

        tokens.append(Token(
            text=outcome_text,
            token_ids=[],
            is_observation=True,
            semantic_type="outcome",
            position=pos,
        ))

        return tokens
