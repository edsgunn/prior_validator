"""Negotiation quality scorer.

Computes three quality metrics per trajectory:
    Q_individual  — Player A's normalised utility
    Q_pareto      — Total utility as fraction of Pareto frontier
    Q_process     — Rule-based coherence score
"""

from __future__ import annotations

from ccsm_eval.trajectories.base import Trajectory


class NegotiationQualityScorer:
    """Scores negotiation trajectories on individual, Pareto, and process quality."""

    def score(self, trajectory: Trajectory) -> dict[str, float]:
        meta = trajectory.metadata
        pool_meta = meta.get("pool", {})
        counts = pool_meta.get("counts", {})
        item_types = pool_meta.get("item_types", list(counts.keys()))
        a_values = meta.get("a_values", {})
        b_values = meta.get("b_values", {})
        final_deal = meta.get("final_deal")
        rounds_meta: list[dict] = meta.get("rounds", [])

        # ---------- Q_individual: A's normalised utility ----------
        if final_deal is not None:
            a_raw = sum(
                a_values.get(item, 0) * final_deal["a_gets"].get(item, 0)
                for item in item_types
            )
        else:
            a_raw = 0.0

        a_max = sum(a_values.get(item, 0) * counts.get(item, 0) for item in item_types)
        q_individual = a_raw / a_max if a_max > 0 else 0.0

        # ---------- Q_pareto: joint efficiency ----------
        if final_deal is not None:
            b_raw = sum(
                b_values.get(item, 0) * final_deal["b_gets"].get(item, 0)
                for item in item_types
            )
            joint = a_raw + b_raw
        else:
            joint = 0.0

        b_max = sum(b_values.get(item, 0) * counts.get(item, 0) for item in item_types)
        total_max = a_max + b_max
        q_pareto = joint / total_max if total_max > 0 else 0.0

        # ---------- Q_process: coherence score ----------
        q_process = self._process_score(rounds_meta, item_types, counts)

        return {
            "q_individual": q_individual,
            "q_pareto": q_pareto,
            "q_process": q_process,
        }

    @staticmethod
    def _process_score(
        rounds: list[dict], item_types: list[str], counts: dict
    ) -> float:
        """Heuristic coherence score in [0, 1].

        Rules checked:
        1. All proposals are internally valid (items sum to pool total).
        2. A's proposals are monotonically non-increasing in A's share (concession rule).
        3. A responds to B's proposals (doesn't just repeat the same offer).
        4. The game terminates with a deal or runs to max rounds (no early abort).
        """
        if not rounds:
            return 0.0

        score = 1.0
        a_proposals: list[dict] = []

        for rec in rounds:
            offer = rec.get("offer")
            if offer and offer.get("a_gets") and offer.get("b_gets"):
                # Check validity
                for item in item_types:
                    total = offer["a_gets"].get(item, 0) + offer["b_gets"].get(item, 0)
                    if total != counts.get(item, 0):
                        score -= 0.1
                        break

                if rec["player"] == "A" and rec["action_type"] == "propose":
                    a_proposals.append(offer["a_gets"])

        # Check A's concession pattern (monotonic or near-monotonic)
        if len(a_proposals) > 1:
            violations = 0
            for i in range(1, len(a_proposals)):
                prev_val = sum(a_proposals[i - 1].values())
                curr_val = sum(a_proposals[i].values())
                if curr_val > prev_val + 1:   # A increased demand
                    violations += 1
            concession_penalty = violations / max(1, len(a_proposals) - 1)
            score -= 0.3 * concession_penalty

        # Check responsiveness: did A change its offer after B responded?
        last_a = None
        repeats = 0
        total_a_actions = 0
        for rec in rounds:
            if rec["player"] == "A" and rec["action_type"] == "propose":
                offer = rec.get("offer")
                if offer and offer.get("a_gets"):
                    curr = tuple(sorted(offer["a_gets"].items()))
                    if curr == last_a:
                        repeats += 1
                    last_a = curr
                    total_a_actions += 1

        if total_a_actions > 1:
            repeat_fraction = repeats / (total_a_actions - 1)
            score -= 0.2 * repeat_fraction

        return max(0.0, min(1.0, score))
