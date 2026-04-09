"""Quality scorer for Concordia trajectories.

Quality scores are computed during generation and stored in trajectory.quality_scores.
This scorer just reads those values back, making it compatible with the run_eval.py
pipeline which calls scorer.score(trajectory) after generating trajectories.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


class ConcordiaQualityScorer:
    """Reads quality scores pre-computed during Concordia episode generation.

    The three universal metrics are:
        payoff           — focal agent's normalised payoff [0, 1]
        social_welfare   — total agent payoffs normalised [0, 1]
        cooperation_score — scenario-specific cooperation measure [0, 1]
    """

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        """Return quality scores. Scores are already in trajectory.quality_scores."""
        scores = dict(trajectory.quality_scores)
        # Ensure all three canonical metrics are present
        for key in ("payoff", "social_welfare", "cooperation_score"):
            scores.setdefault(key, 0.0)
        return scores
