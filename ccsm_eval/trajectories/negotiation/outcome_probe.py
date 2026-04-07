"""Outcome surprise probe for negotiation trajectories.

Appends an outcome summary observation token and measures the model's
surprise at it — testing whether the model has internalised the consequences
of negotiation strategies.
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Token, Trajectory


class OutcomeSurpriseProbe:
    """Appends an outcome summary token to a negotiation trajectory.

    The summary describes the final score as a fraction of the maximum
    achievable. After calling `augment`, the returned trajectory has one
    extra observation token at the end of type "outcome_probe".

    This is used in the SurpriseEvaluator to measure:
        "Does the model find a high score surprising after a bad negotiation?"
    """

    def augment(self, trajectory: Trajectory) -> Trajectory:
        """Return a copy of the trajectory with the outcome probe token appended."""
        import copy
        traj = copy.deepcopy(trajectory)

        a_util = traj.metadata.get("a_utility_normalised", 0.0)
        pareto = traj.metadata.get("pareto_efficiency", 0.0)
        final_outcome = traj.metadata.get("final_outcome", "disagreement")

        if "deal" in final_outcome:
            score_text = f"{a_util * 10:.0f} out of 10 possible points"
            probe_text = f"[Outcome: Deal reached. You scored {score_text}.]\n"
        else:
            probe_text = "[Outcome: No deal reached. You scored 0 out of 10 possible points.]\n"

        next_pos = (traj.tokens[-1].position + 1) if traj.tokens else 0
        probe_token = Token(
            text=probe_text,
            token_ids=[],
            is_observation=True,
            semantic_type="outcome_probe",
            position=next_pos,
        )
        traj.tokens.append(probe_token)

        # Stash probe metadata
        traj.metadata["outcome_probe_text"] = probe_text
        traj.metadata["outcome_probe_a_util"] = a_util
        traj.metadata["outcome_probe_pareto"] = pareto

        return traj

    def is_plausible(self, trajectory: Trajectory) -> bool:
        """Check if the outcome probe token is coherent with the trajectory."""
        final_outcome = trajectory.metadata.get("final_outcome", "disagreement")
        a_util = trajectory.metadata.get("a_utility_normalised", 0.0)

        if "deal" in final_outcome:
            return a_util > 0.0
        return a_util == 0.0
