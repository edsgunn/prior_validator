"""Chess trajectory formatter: FEN+algebraic and natural language variants.

The formatter converts raw Trajectory objects (as produced by
ChessTrajectoryGenerator) into a text sequence suitable for model input,
plus a character-level observation mask and semantic type map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import chess

from ccsm_eval.trajectories.base import Trajectory


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TrajectoryFormatter(ABC):
    @abstractmethod
    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        """Return (full_text, char_observation_mask, char_semantic_type).

        full_text               — the complete string to feed the model.
        char_observation_mask   — per-character bool: True if the character
                                  belongs to an observation token (σ_t = 1).
        char_semantic_type      — per-character semantic type label.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FEN + algebraic notation formatter
# ---------------------------------------------------------------------------

class FENFormatter(TrajectoryFormatter):
    """Compact FEN + SAN algebraic notation.

    Format:
        <character_prompt>\\n
        Position: <initial_FEN>\\n
        White: <SAN_move>\\n
        Black: <SAN_move>  Position: <FEN>\\n
        White: <SAN_move>\\n
        ...
    """

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        # Prompt (neither observation nor action — treat as non-observation)
        prompt_block = character_prompt.strip() + "\n"
        _append(text_parts, mask_parts, type_parts, prompt_block, False, "prompt")

        records: list[dict] = trajectory.metadata.get("move_records", [])

        if not records:
            return _assemble(text_parts, mask_parts, type_parts)

        # Initial board state
        initial_fen = records[0]["fen_before"]
        obs_text = f"Position: {initial_fen}\n"
        _append(text_parts, mask_parts, type_parts, obs_text, True, "board_state")

        i = 0
        while i < len(records):
            record = records[i]
            if record["color"] == "white":
                # White's action
                act_text = f"White: {record['san']}\n"
                _append(text_parts, mask_parts, type_parts, act_text, False, "agent_move")

                # Black's response observation
                if i + 1 < len(records) and records[i + 1]["color"] == "black":
                    black = records[i + 1]
                    black_move_text = f"Black: {black['san']}  "
                    _append(
                        text_parts, mask_parts, type_parts,
                        black_move_text, True, "opponent_move"
                    )
                    pos_text = f"Position: {black['fen_after']}\n"
                    _append(
                        text_parts, mask_parts, type_parts,
                        pos_text, True, "board_state"
                    )
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Game result
        outcome = trajectory.metadata.get("outcome", "*")
        if outcome != "*":
            result_text = f"Result: {outcome}\n"
            _append(text_parts, mask_parts, type_parts, result_text, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)


# ---------------------------------------------------------------------------
# Natural language narration formatter
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}

_FILE_NAMES = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"
}

_RANK_NAMES = {i: str(i + 1) for i in range(8)}

_OUTCOME_PHRASES = {
    "1-0": "White wins.",
    "0-1": "Black wins.",
    "1/2-1/2": "The game ends in a draw.",
    "*": "The game is ongoing.",
}


def _square_name(sq: chess.Square) -> str:
    return f"{_FILE_NAMES[chess.square_file(sq)]}{_RANK_NAMES[chess.square_rank(sq)]}"


def _narrate_move(board_before: chess.Board, move: chess.Move, color: str) -> str:
    """Produce a short natural language description of a chess move."""
    piece = board_before.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(piece.symbol().upper(), "piece") if piece else "piece"

    from_sq = _square_name(move.from_square)
    to_sq = _square_name(move.to_square)

    is_capture = board_before.is_capture(move)
    is_castle = board_before.is_castling(move)
    is_promotion = move.promotion is not None

    if is_castle:
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            return f"{color.capitalize()} castles kingside."
        else:
            return f"{color.capitalize()} castles queenside."

    if is_capture:
        captured = board_before.piece_at(move.to_square)
        cap_name = _PIECE_NAMES.get(captured.symbol().upper(), "piece") if captured else "piece"
        desc = f"{color.capitalize()} captures the {cap_name} on {to_sq} with the {piece_name} from {from_sq}."
    else:
        desc = f"{color.capitalize()} moves the {piece_name} from {from_sq} to {to_sq}."

    if is_promotion:
        promo_name = _PIECE_NAMES.get(
            chess.piece_name(move.promotion).upper()[0], "queen"
        )
        desc = desc.rstrip(".") + f", promoting to a {promo_name}."

    board_after = board_before.copy()
    board_after.push(move)
    if board_after.is_checkmate():
        desc = desc.rstrip(".") + " Checkmate!"
    elif board_after.is_check():
        desc = desc.rstrip(".") + " Check."

    return desc


class NaturalLanguageFormatter(TrajectoryFormatter):
    """Narrates the chess game in natural language prose.

    Format:
        <character_prompt>\\n
        The game begins. [brief position description]\\n
        White moves the knight from g1 to f3. ...\\n
        Black responds: Black moves the pawn from e7 to e5. ...\\n
        ...
    """

    def format(
        self, trajectory: Trajectory, character_prompt: str
    ) -> tuple[str, list[bool], list[str]]:
        text_parts: list[str] = []
        mask_parts: list[list[bool]] = []
        type_parts: list[list[str]] = []

        prompt_block = character_prompt.strip() + "\n"
        _append(text_parts, mask_parts, type_parts, prompt_block, False, "prompt")

        records: list[dict] = trajectory.metadata.get("move_records", [])

        if not records:
            return _assemble(text_parts, mask_parts, type_parts)

        # Opening observation
        obs_text = "The game begins from the standard starting position.\n"
        _append(text_parts, mask_parts, type_parts, obs_text, True, "board_state")

        board = chess.Board()
        i = 0
        while i < len(records):
            record = records[i]
            move = chess.Move.from_uci(record["uci"])

            if record["color"] == "white":
                narration = _narrate_move(board, move, "white")
                act_text = narration + "\n"
                _append(text_parts, mask_parts, type_parts, act_text, False, "agent_move")
                board.push(move)

                if i + 1 < len(records) and records[i + 1]["color"] == "black":
                    black_rec = records[i + 1]
                    black_move = chess.Move.from_uci(black_rec["uci"])
                    black_narration = _narrate_move(board, black_move, "black")
                    obs_text = black_narration + "\n"
                    _append(
                        text_parts, mask_parts, type_parts,
                        obs_text, True, "opponent_move"
                    )
                    board.push(black_move)
                    i += 2
                else:
                    i += 1
            else:
                move = chess.Move.from_uci(record["uci"])
                board.push(move)
                i += 1

        outcome = trajectory.metadata.get("outcome", "*")
        result_text = _OUTCOME_PHRASES.get(outcome, "") + "\n"
        if result_text.strip():
            _append(text_parts, mask_parts, type_parts, result_text, True, "outcome")

        return _assemble(text_parts, mask_parts, type_parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_chess_formatter(fmt: str) -> TrajectoryFormatter:
    """Return the requested chess formatter.

    Args:
        fmt: "fen" or "natural"
    """
    if fmt == "fen":
        return FENFormatter()
    elif fmt == "natural":
        return NaturalLanguageFormatter()
    else:
        raise ValueError(f"Unknown chess format: {fmt!r}. Choose 'fen' or 'natural'.")
