# outcome_model.py
"""
Bootstrap ensemble of outcome prediction models.

Each ensemble member is a logistic regression for P(MSMI | pair features).
Members start from different random priors and are refit every day on a
bootstrap resample of the observed (pair, feedback) data, so their
disagreement measures uncertainty (used by Thompson sampling).

All randomness uses local random.Random instances, so this module never
touches Python's global random state (which the simulator relies on).
"""

import math
import random

FEATURES = ['age_diff', 'gender_match', 'goal_compat',
            'lifestyle_compat', 'zone_overlap', 'smoking_compat']


class LogisticRegressionModel:
    """Logistic regression for P(MSMI | features)."""

    def __init__(self, seed=42):
        self.seed = seed
        rng = random.Random(seed)
        self.weights = {
            'age_diff': rng.gauss(-0.5, 0.2),
            'gender_match': rng.gauss(0.3, 0.3),
            'goal_compat': rng.gauss(0.3, 0.3),
            'lifestyle_compat': rng.gauss(0.3, 0.3),
            'zone_overlap': rng.gauss(0.3, 0.3),
            'smoking_compat': rng.gauss(0.3, 0.3),
            'intercept': rng.gauss(0.0, 0.3),
        }
        # Prior = initial weights; fitting is regularised towards it.
        self.prior = dict(self.weights)

    # ------------------------------------------------------------------
    def extract_features(self, person_a, person_b):
        """Feature dict, all roughly in [0, 1]."""
        age_diff = min(abs(person_a.get('age', 30) - person_b.get('age', 30)) / 40.0, 1.0)

        gender_match = 1.0 if person_a.get('gender', 'M') != person_b.get('gender', 'F') else 0.0

        a_goal, b_goal = person_a.get('relationship_goal'), person_b.get('relationship_goal')
        goal_compat = (1.0 if a_goal == b_goal else 0.0) if (a_goal and b_goal) else 0.5

        a_life, b_life = person_a.get('lifestyle'), person_b.get('lifestyle')
        lifestyle_compat = (1.0 if a_life == b_life else 0.0) if (a_life and b_life) else 0.5

        a_zone = person_a.get('zone', 'downtown')
        b_zone = person_b.get('zone', 'downtown')
        a_ok = person_a.get('acceptable_zones', [])
        b_ok = person_b.get('acceptable_zones', [])
        if a_zone == b_zone:
            zone_compat = 1.0
        elif (a_ok and b_zone in a_ok) or (b_ok and a_zone in b_ok):
            zone_compat = 0.5
        else:
            zone_compat = 0.0

        # Smoking: check both directions; unknown = neutral 0.5
        checks = []
        for p, q in ((person_a, person_b), (person_b, person_a)):
            want, has = p.get('partner_smoking'), q.get('smoking')
            if want is not None and has is not None:
                checks.append(1.0 if want == has else 0.0)
        smoking_compat = sum(checks) / len(checks) if checks else 0.5

        return {
            'age_diff': age_diff,
            'gender_match': gender_match,
            'goal_compat': goal_compat,
            'lifestyle_compat': lifestyle_compat,
            'zone_overlap': zone_compat,
            'smoking_compat': smoking_compat,
        }

    def _predict_from_vector(self, x):
        z = self.weights['intercept']
        for name, xi in zip(FEATURES, x):
            z += self.weights[name] * xi
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_msmi(self, person_a, person_b):
        feats = self.extract_features(person_a, person_b)
        return self._predict_from_vector([feats[k] for k in FEATURES])

    # ------------------------------------------------------------------
    def fit(self, X, y, epochs=20, lr=0.8, l2=0.05):
        """
        Batch gradient descent on log-loss, with an L2 pull towards the prior
        weights (so a few observations do not swing the model wildly).

        X: list of feature vectors (ordered as FEATURES); y: list of 0/1.
        """
        n = len(X)
        if n == 0:
            return
        names = FEATURES
        w = [self.weights[k] for k in names]
        b = self.weights['intercept']
        w0 = [self.prior[k] for k in names]
        b0 = self.prior['intercept']

        for _ in range(epochs):
            gw = [0.0] * len(w)
            gb = 0.0
            for xi, yi in zip(X, y):
                z = b
                for j in range(len(w)):
                    z += w[j] * xi[j]
                z = max(-50.0, min(50.0, z))
                err = 1.0 / (1.0 + math.exp(-z)) - yi
                gb += err
                for j in range(len(w)):
                    gw[j] += err * xi[j]
            for j in range(len(w)):
                w[j] -= lr * (gw[j] / n + l2 * (w[j] - w0[j]))
            b -= lr * (gb / n + l2 * (b - b0))

        for k, wj in zip(names, w):
            self.weights[k] = wj
        self.weights['intercept'] = b


class OutcomeEnsemble:
    """Bootstrap ensemble of LogisticRegressionModel."""

    def __init__(self, n_ensemble=10, seed=42, max_obs=400, min_obs=20):
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.max_obs = max_obs      # only the most recent max_obs outcomes are used
        self.min_obs = min_obs      # do not fit before this many outcomes exist
        self.models = [LogisticRegressionModel(seed=seed + i) for i in range(n_ensemble)]
        self.trained_counts = [0] * n_ensemble
        self._last_n = 0

    def train(self, introductions, feedback, members_by_id=None):
        """
        Refit every member on a bootstrap resample of observed outcomes.

        introductions: list of (a_id, b_id, day)
        feedback:      dict {(a_id, b_id): bool}
        members_by_id: dict {member_id: member dict}; required to build features
        """
        if not introductions or members_by_id is None:
            return

        recent = introductions[-self.max_obs:] if self.max_obs else introductions
        obs = []
        for a_id, b_id, _day in recent:
            if (a_id, b_id) not in feedback:
                continue
            a, b = members_by_id.get(a_id), members_by_id.get(b_id)
            if not a or not b:
                continue
            feats = self.models[0].extract_features(a, b)
            obs.append(([feats[k] for k in FEATURES], 1.0 if feedback[(a_id, b_id)] else 0.0))

        n = len(obs)
        if n < self.min_obs or n == self._last_n:
            return
        self._last_n = n

        for i, model in enumerate(self.models):
            rng = random.Random(self.seed * 1000 + i * 17 + n)
            idx = [rng.randrange(n) for _ in range(n)]       # bootstrap resample
            model.fit([obs[j][0] for j in idx], [obs[j][1] for j in idx])
            self.trained_counts[i] = n

    # ------------------------------------------------------------------
    def predict_msmi(self, person_a, person_b, ensemble_idx=0):
        if not person_a or not person_b:
            return 0.5
        return self.models[ensemble_idx % self.n_ensemble].predict_msmi(person_a, person_b)

    def predict_msmi_ensemble(self, person_a, person_b):
        """Return (mean, std) across all members."""
        if not person_a or not person_b:
            return 0.5, 0.0
        preds = [m.predict_msmi(person_a, person_b) for m in self.models]
        mean = sum(preds) / len(preds)
        if len(preds) > 1:
            var = sum((p - mean) ** 2 for p in preds) / (len(preds) - 1)
            std = math.sqrt(max(0.0, var))
        else:
            std = 0.0
        return mean, std

    def get_uncertainty(self, person_a, person_b):
        return self.predict_msmi_ensemble(person_a, person_b)[1]
