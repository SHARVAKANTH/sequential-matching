# thompson_sampling.py
"""
Thompson sampling over the bootstrap ensemble.

Each day: sample one ensemble member as the "true world", score all feasible
pairs with it, and act greedily on those scores. Disagreement between members
produces exploration; as data accumulates the members converge and exploration
shrinks on its own.

Uses a local random.Random so the global random state (used by the simulator
to draw feedback) is never reseeded.
"""

import random


class ThompsonSampler:
    def __init__(self, ensemble, seed=42):
        self.ensemble = ensemble
        self.seed = seed
        self.rng = random.Random(seed + 7919)
        self.sampled_idx = 0
        self.day = 0
        self.day_history = []

    def sample_world(self):
        """Pick one ensemble member uniformly for today."""
        self.sampled_idx = self.rng.randrange(self.ensemble.n_ensemble)
        self.day_history.append(self.sampled_idx)
        return self.sampled_idx

    def score_all_pairs(self, members, mask, ensemble_idx):
        """
        Score every pair with one ensemble member.

        Infeasible pairs (mask == 0) get score 0.0 so they can never be
        chosen by the matcher (which skips score <= 0).

        Returns: {(a_id, b_id): score in [0, 1]} with a_id < b_id.
        """
        scores = {}
        for i in range(len(members)):
            a = members[i]
            for j in range(i + 1, len(members)):
                b = members[j]
                pair = tuple(sorted((a['member_id'], b['member_id'])))
                if mask.get(pair, 0) == 0:
                    scores[pair] = 0.0
                    continue
                s = self.ensemble.predict_msmi(a, b, ensemble_idx)
                scores[pair] = max(0.0, min(1.0, s))
        return scores

    def advance_day(self):
        self.day += 1

    def get_day_history(self):
        return self.day_history.copy()
