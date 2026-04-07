"""Chess quality scoring.

Reads centipawn data pre-computed by the generator (stored in move_records).
No Stockfish calls needed at score time.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


class ChessQualityScorer:
    """Computes quality metrics for chess trajectories.

    Metrics are derived from `best_cp` / `move_cp` values written into
    move_records by ChessTrajectoryGenerator during game play — no second
    Stockfish pass needed.

    Metrics:
        centipawn_eval  — mean position score (from White's POV) after each
                          of White's moves.
        mean_cp_loss    — mean(best_cp - move_cp) across White's moves.
        outcome         — game result from White's perspective: +1 win, 0 draw,
                          -1 loss.
    """

    def __init__(self, stockfish_path: str = "stockfish", eval_time: float = 0.05):
        # stockfish_path / eval_time kept for API compatibility; not used.
        pass

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        move_records: list[dict] = trajectory.metadata.get("move_records", [])
        outcome_str: str = trajectory.metadata.get("outcome", "*")
        outcome = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0, "*": 0.0}[outcome_str]

        cp_evals: list[float] = []
        cp_losses: list[float] = []

        for record in move_records:
            if record["color"] != "white":
                continue
            best_cp = record.get("best_cp")
            move_cp = record.get("move_cp")
            if best_cp is not None and move_cp is not None:
                cp_evals.append(move_cp)
                cp_losses.append(max(0.0, best_cp - move_cp))

        mean_eval = sum(cp_evals) / len(cp_evals) if cp_evals else 0.0
        mean_loss = sum(cp_losses) / len(cp_losses) if cp_losses else 0.0

        return {
            "centipawn_eval": mean_eval,
            "mean_cp_loss": mean_loss,
            "outcome": outcome,
        }
