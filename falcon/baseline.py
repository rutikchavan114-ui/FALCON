from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


class ReferenceBaseline:
    def __init__(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError("Expected an (N, 2) feature matrix.")
        self.mean = features.mean(axis=0)
        self.cov_estimator = LedoitWolf().fit(features)
        self.covariance = self.cov_estimator.covariance_
        self.precision = self.cov_estimator.precision_

    def score(self, feature):
        d = np.asarray(feature, dtype=float) - self.mean
        return float(d @ self.precision @ d)
