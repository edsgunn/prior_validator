"""Chess trajectory generator using python-chess and Stockfish.

White's moves are selected according to a quality level. Black always plays
Stockfish at a fixed depth to provide a consistent environment.
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import chess
import chess.engine

from ccsm_eval.trajectories.base import Token, Trajectory


# ---------------------------------------------------------------------------
# Abstract base for trajectory generators
# ---------------------------------------------------------------------------

class TrajectoryGenerator(ABC):
    @abstractmethod
    def generate(
        self, quality_level: str, n_trajectories: int, seed: int
    ) -> list[Trajectory]:
        ...

    @abstractmethod
    def quality_levels(self) -> list[str]:
        ...


# ---------------------------------------------------------------------------
# Chess trajectory generator
# ---------------------------------------------------------------------------

QUALITY_LEVELS = ["optimal", "strong", "moderate", "weak", "random"]

_MATE_CP = 10_000


def _score_to_cp(score: chess.engine.PovScore) -> float:
    if score.is_mate():
        return float(_MATE_CP) if score.mate() > 0 else float(-_MATE_CP)
    return float(score.score())


class ChessTrajectoryGenerator(TrajectoryGenerator):
    """Generates chess trajectories where White plays at a controlled quality level.

    White's move selection:
        optimal  — Stockfish top-1 at white_depth (default 20)
        strong   — Stockfish top-3 (sampled uniformly)
        moderate — Stockfish top-10 (sampled uniformly)
        weak     — Stockfish bottom-half legal moves (sampled)
        random   — Uniform random legal move

    Black always plays Stockfish at opponent_depth (default 15).
    """

    def __init__(
        self,
        stockfish_path: str = "stockfish",
        white_time: float = 0.1,    # Seconds per move for White's analysis
        opponent_time: float = 0.05, # Seconds per move for Black
        max_moves: int = 120,       # Half-moves (plies)
        multipv_cache: int = 20,    # How many moves to request from Stockfish
    ):
        self.stockfish_path = stockfish_path
        self.white_time = white_time
        self.opponent_time = opponent_time
        self.max_moves = max_moves
        self.multipv_cache = multipv_cache

    def quality_levels(self) -> list[str]:
        return QUALITY_LEVELS

    def generate(
        self, quality_level: str, n_trajectories: int, seed: int
    ) -> list[Trajectory]:
        if quality_level not in QUALITY_LEVELS:
            raise ValueError(f"Unknown quality level: {quality_level!r}")

        rng = random.Random(seed)
        trajectories = []

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            for i in range(n_trajectories):
                traj_seed = rng.randint(0, 2**31)
                traj = self._generate_one(engine, quality_level, traj_seed)
                trajectories.append(traj)

        return trajectories

    def _generate_one(
        self,
        engine: chess.engine.SimpleEngine,
        quality_level: str,
        seed: int,
    ) -> Trajectory:
        rng = random.Random(seed)
        board = chess.Board()

        # Collect raw game data: list of (move_uci, fen_before, fen_after)
        move_records: list[dict] = []

        for ply in range(self.max_moves):
            if board.is_game_over():
                break

            best_cp: Optional[float] = None
            move_cp: Optional[float] = None

            if board.turn == chess.WHITE:
                move, best_cp, move_cp = self._select_white_move(
                    engine, board, quality_level, rng
                )
            else:
                move = self._select_black_move(engine, board)

            if move is None:
                break

            fen_before = board.fen()
            san = board.san(move)
            board.push(move)
            fen_after = board.fen()

            record: dict = {
                "uci": move.uci(),
                "san": san,
                "fen_before": fen_before,
                "fen_after": fen_after,
                "color": "white" if ply % 2 == 0 else "black",
            }
            if best_cp is not None:
                record["best_cp"] = best_cp
                record["move_cp"] = move_cp
            move_records.append(record)

        outcome = board.result()  # "1-0", "0-1", "1/2-1/2", or "*"

        # Build Token list (no tokenisation yet — that's the formatter's job)
        tokens = self._build_tokens(move_records)

        traj_id = str(uuid.uuid4())
        return Trajectory(
            trajectory_id=traj_id,
            tokens=tokens,
            character_prompt="",   # Filled in by the runner per prompt config
            quality_scores={},     # Filled in by ChessQualityScorer
            quality_level=quality_level,
            environment="chess",
            metadata={
                "moves": [r["uci"] for r in move_records],
                "move_records": move_records,
                "outcome": outcome,
                "final_fen": board.fen(),
            },
        )

    def _build_tokens(self, move_records: list[dict]) -> list[Token]:
        """Build the raw token list from move records.

        Token layout:
            [obs]  Initial board state (FEN)
            [act]  White's move (SAN)
            [obs]  Black's response + resulting position (FEN)
            [act]  White's move
            [obs]  Black's response + resulting position
            ...

        The actual text representation is finalised by ChessFormatter, which
        knows what format (FEN vs natural language) to use. Here we store the
        structured data and set is_observation / semantic_type.
        """
        tokens: list[Token] = []
        position = 0

        if not move_records:
            return tokens

        # Initial board state (observation)
        tokens.append(
            Token(
                text=move_records[0]["fen_before"],
                token_ids=[],
                is_observation=True,
                semantic_type="board_state",
                position=position,
            )
        )
        position += 1

        i = 0
        while i < len(move_records):
            record = move_records[i]

            if record["color"] == "white":
                # White's action token
                tokens.append(
                    Token(
                        text=record["san"],
                        token_ids=[],
                        is_observation=False,
                        semantic_type="agent_move",
                        position=position,
                    )
                )
                position += 1

                # Observation: Black's response + resulting position
                if i + 1 < len(move_records) and move_records[i + 1]["color"] == "black":
                    black_record = move_records[i + 1]
                    # Two sub-tokens combined into one observation block
                    tokens.append(
                        Token(
                            text=black_record["san"],
                            token_ids=[],
                            is_observation=True,
                            semantic_type="opponent_move",
                            position=position,
                        )
                    )
                    position += 1
                    tokens.append(
                        Token(
                            text=black_record["fen_after"],
                            token_ids=[],
                            is_observation=True,
                            semantic_type="board_state",
                            position=position,
                        )
                    )
                    position += 1
                    i += 2
                else:
                    i += 1
            else:
                # Stray black move (shouldn't happen with even starts)
                i += 1

        return tokens

    # ------------------------------------------------------------------
    # Move selection helpers
    # ------------------------------------------------------------------

    def _select_white_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        quality_level: str,
        rng: random.Random,
    ) -> tuple[Optional[chess.Move], Optional[float], Optional[float]]:
        """Return (move, best_cp, move_cp). cp values are None for random."""
        legal = list(board.legal_moves)
        if not legal:
            return None, None, None

        if quality_level == "random":
            return rng.choice(legal), None, None

        if quality_level == "weak":
            return self._select_weak_move(engine, board, legal, rng)

        # For optimal/strong/moderate, query Stockfish multipv
        n_pv = {"optimal": 1, "strong": 3, "moderate": 10}[quality_level]
        n_pv = min(n_pv, len(legal))

        info_list = engine.analyse(
            board,
            chess.engine.Limit(time=self.white_time),
            multipv=n_pv,
        )
        # info_list is sorted best-first; score is position value after playing PV[0]
        best_cp = _score_to_cp(info_list[0]["score"].white())
        chosen_info = rng.choice(info_list)
        move_cp = _score_to_cp(chosen_info["score"].white())
        pv = chosen_info.get("pv")
        move = pv[0] if pv else rng.choice(legal)
        return move, best_cp, move_cp

    def _select_weak_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        legal: list[chess.Move],
        rng: random.Random,
    ) -> tuple[chess.Move, float, float]:
        """Select from the bottom half of legal moves by Stockfish score."""
        n = max(1, len(legal))
        n_pv = min(self.multipv_cache, n)

        info_list = engine.analyse(
            board,
            chess.engine.Limit(time=self.white_time),
            multipv=n_pv,
        )
        # Sorted best-first; take the bottom half
        best_cp = _score_to_cp(info_list[0]["score"].white())
        bottom = info_list[len(info_list) // 2 :]
        if not bottom:
            bottom = info_list
        chosen = rng.choice(bottom)
        move_cp = _score_to_cp(chosen["score"].white())
        pv = chosen.get("pv")
        move = pv[0] if pv else rng.choice(legal)
        return move, best_cp, move_cp

    def _select_black_move(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
    ) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        if not legal:
            return None
        result = engine.play(board, chess.engine.Limit(time=self.opponent_time))
        return result.move
