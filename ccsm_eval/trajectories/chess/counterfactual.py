"""Counterfactual editor for chess trajectories.

Replaces a single White move with a better or worse alternative,
then replays the game from that point with Black responding via Stockfish.
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod

import chess
import chess.engine

from ccsm_eval.trajectories.base import CounterfactualEdit, Token, Trajectory

# Minimum centipawn difference to qualify as a meaningful edit
DEFAULT_MIN_CP_DELTA = 100


class CounterfactualEditor(ABC):
    @abstractmethod
    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        ...

    @abstractmethod
    def sample_edit_positions(
        self, trajectory: Trajectory, n_edits: int, seed: int
    ) -> list[int]:
        ...


class ChessCounterfactualEditor(CounterfactualEditor):
    """Replaces a White move with a better ('up') or worse ('down') alternative.

    The edit proceeds by:
    1. Rebuilding the board up to the target ply.
    2. Selecting a replacement move at least `min_cp_delta` centipawns
       different from the original.
    3. Replaying one move-pair (White replacement + Black Stockfish response).
    4. Returning the observation tokens for both the original and replacement
       sequences.
    """

    def __init__(
        self,
        stockfish_path: str = "stockfish",
        eval_time: float = 0.05,
        opponent_time: float = 0.05,
        min_cp_delta: int = DEFAULT_MIN_CP_DELTA,
        n_followup_plies: int = 2,   # How many plies after edit to include in tokens
    ):
        self.stockfish_path = stockfish_path
        self.eval_time = eval_time
        self.opponent_time = opponent_time
        self.min_cp_delta = min_cp_delta
        self.n_followup_plies = n_followup_plies

    def quality_levels(self) -> list[str]:  # satisfies duck-typing used in run_eval
        return []

    def sample_edit_positions(
        self, trajectory: Trajectory, n_edits: int, seed: int
    ) -> list[int]:
        """Return indices into trajectory.metadata["move_records"] for White plies."""
        records: list[dict] = trajectory.metadata.get("move_records", [])
        white_indices = [i for i, r in enumerate(records) if r["color"] == "white"]
        # Skip first and last to ensure at least one Black reply exists
        white_indices = white_indices[1:-1] if len(white_indices) > 2 else white_indices
        rng = random.Random(seed)
        n = min(n_edits, len(white_indices))
        return rng.sample(white_indices, n)

    def edit(
        self, trajectory: Trajectory, position: int, direction: str
    ) -> CounterfactualEdit:
        """Create a counterfactual edit at the given move-record index.

        Args:
            trajectory: The original trajectory.
            position:   Index into metadata["move_records"] (must be White's turn).
            direction:  "up" (replace with a better move) or "down" (worse).

        Returns:
            CounterfactualEdit with original and replacement observation tokens.
        """
        if direction not in ("up", "down"):
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        records: list[dict] = trajectory.metadata.get("move_records", [])
        if position >= len(records) or records[position]["color"] != "white":
            raise ValueError(
                f"Position {position} is not a White move in this trajectory."
            )

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            # Replay to the position just before the target White move
            board = chess.Board()
            for i in range(position):
                board.push(chess.Move.from_uci(records[i]["uci"]))

            original_uci = records[position]["uci"]
            original_move = chess.Move.from_uci(original_uci)

            # Score the original move
            orig_best_cp, orig_move_cp = self._score_move(engine, board, original_move)
            original_cp_loss = orig_best_cp - orig_move_cp

            # Find a replacement
            replacement_move, quality_delta = self._find_replacement(
                engine, board, original_move, orig_move_cp, direction
            )

            if replacement_move is None:
                # Fallback: use original (delta ~ 0)
                replacement_move = original_move
                quality_delta = 0.0

            # Build observation tokens for the original sequence
            original_obs = self._replay_observation(
                engine, board.copy(), original_move
            )

            # Build observation tokens for the replacement sequence
            replacement_obs = self._replay_observation(
                engine, board.copy(), replacement_move
            )

        return CounterfactualEdit(
            trajectory_id=trajectory.trajectory_id,
            edit_position=position,
            original_action=board.san(original_move)
            if original_move in board.legal_moves
            else original_uci,
            replacement_action=board.san(replacement_move)
            if replacement_move in board.legal_moves
            else replacement_move.uci(),
            direction=direction,
            quality_delta=quality_delta,
            original_tokens=original_obs,
            replacement_tokens=replacement_obs,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        move: chess.Move,
    ) -> tuple[float, float]:
        """Return (best_cp_before, move_cp_after) from White's perspective."""
        limit = chess.engine.Limit(time=self.eval_time)
        info = engine.analyse(board, limit, multipv=1)
        best_cp = self._to_cp(info[0]["score"].white())

        b2 = board.copy()
        b2.push(move)
        info2 = engine.analyse(b2, limit, multipv=1)
        move_cp = self._to_cp(info2[0]["score"].white())

        return best_cp, move_cp

    def _find_replacement(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        original_move: chess.Move,
        original_cp: float,
        direction: str,
    ) -> tuple[chess.Move | None, float]:
        """Find a move that is at least min_cp_delta better/worse than the original."""
        legal = list(board.legal_moves)
        n_pv = min(len(legal), 20)
        limit = chess.engine.Limit(time=self.eval_time)
        info_list = engine.analyse(board, limit, multipv=n_pv)

        best_move: chess.Move | None = None
        best_delta = 0.0

        for info in info_list:
            pv = info.get("pv")
            if not pv:
                continue
            candidate = pv[0]
            if candidate == original_move:
                continue

            b2 = board.copy()
            b2.push(candidate)
            cand_info = engine.analyse(b2, chess.engine.Limit(time=self.eval_time))
            cand_cp = self._to_cp(cand_info["score"].white())

            delta = cand_cp - original_cp  # positive = better for White

            if direction == "up" and delta >= self.min_cp_delta:
                if delta > best_delta:
                    best_delta = delta
                    best_move = candidate
            elif direction == "down" and delta <= -self.min_cp_delta:
                if delta < best_delta:
                    best_delta = delta
                    best_move = candidate

        return best_move, best_delta

    def _replay_observation(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        white_move: chess.Move,
    ) -> list[Token]:
        """Play white_move then let Black respond; return the observation tokens."""
        tokens: list[Token] = []
        position = 0

        # White's move produces a board state observation
        white_san = board.san(white_move)
        board.push(white_move)

        fen_after_white = board.fen()
        tokens.append(
            Token(
                text=fen_after_white,
                token_ids=[],
                is_observation=True,
                semantic_type="board_state",
                position=position,
            )
        )
        position += 1

        # Black's response
        if not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=self.opponent_time))
            black_move = result.move
            if black_move:
                black_san = board.san(black_move)
                tokens.append(
                    Token(
                        text=black_san,
                        token_ids=[],
                        is_observation=True,
                        semantic_type="opponent_move",
                        position=position,
                    )
                )
                position += 1
                board.push(black_move)
                tokens.append(
                    Token(
                        text=board.fen(),
                        token_ids=[],
                        is_observation=True,
                        semantic_type="board_state",
                        position=position,
                    )
                )
                position += 1

        return tokens

    @staticmethod
    def _to_cp(score) -> float:
        if score.is_mate():
            return 10_000.0 if score.mate() > 0 else -10_000.0
        return float(score.score())
