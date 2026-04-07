"""Chess quality scoring using Stockfish centipawn evaluation."""

from __future__ import annotations

import chess
import chess.engine

from ccsm_eval.trajectories.base import Trajectory

# Centipawn cap for mate scores
_MATE_CP = 10_000


class ChessQualityScorer:
    """Computes quality metrics for chess trajectories via Stockfish.

    Metrics:
        centipawn_eval  — mean Stockfish centipawn score of White's positions
                          after each of White's moves, from White's perspective.
        mean_cp_loss    — mean centipawn loss of White's moves relative to
                          Stockfish's top-1 choice.
        outcome         — game result from White's perspective: +1 win, 0 draw,
                          -1 loss.
    """

    def __init__(self, stockfish_path: str = "stockfish", eval_depth: int = 15):
        self.stockfish_path = stockfish_path
        self.eval_depth = eval_depth

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        """Return quality scores for a chess trajectory.

        The trajectory metadata must contain:
            "moves"   — list of move strings (full game in SAN or UCI)
            "outcome" — "1-0", "0-1", "1/2-1/2", or "*"
        """
        moves: list[str] = trajectory.metadata.get("moves", [])
        outcome_str: str = trajectory.metadata.get("outcome", "*")

        outcome = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0, "*": 0.0}[outcome_str]

        if not moves:
            return {"centipawn_eval": 0.0, "mean_cp_loss": 0.0, "outcome": outcome}

        board = chess.Board()
        cp_evals: list[float] = []
        cp_losses: list[float] = []

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            limit = chess.engine.Limit(depth=self.eval_depth)

            for i, move_uci in enumerate(moves):
                move = chess.Move.from_uci(move_uci)

                if board.turn == chess.WHITE:
                    # Evaluate best move before making this one
                    best_info = engine.analyse(board, limit, multipv=1)
                    best_score = self._cp(best_info[0]["score"].white())

                    board.push(move)

                    # Evaluate position after White's move
                    pos_info = engine.analyse(board, limit, multipv=1)
                    pos_score = self._cp(pos_info[0]["score"].white())

                    cp_evals.append(pos_score)
                    cp_losses.append(max(0.0, best_score - pos_score))
                else:
                    board.push(move)

        mean_eval = sum(cp_evals) / len(cp_evals) if cp_evals else 0.0
        mean_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0.0

        return {
            "centipawn_eval": mean_eval,
            "mean_cp_loss": mean_loss,
            "outcome": outcome,
        }

    def score_move(
        self, board: chess.Board, move: chess.Move, engine: chess.engine.SimpleEngine
    ) -> tuple[float, float]:
        """Return (best_cp, move_cp) for a single White move on the given board."""
        limit = chess.engine.Limit(depth=self.eval_depth)
        best_info = engine.analyse(board, limit, multipv=1)
        best_cp = self._cp(best_info[0]["score"].white())

        board_copy = board.copy()
        board_copy.push(move)
        move_info = engine.analyse(board_copy, limit, multipv=1)
        move_cp = self._cp(move_info[0]["score"].white())
        return best_cp, move_cp

    @staticmethod
    def _cp(score: chess.engine.PovScore) -> float:
        if score.is_mate():
            return float(_MATE_CP) if score.mate() > 0 else float(-_MATE_CP)
        return float(score.score())
