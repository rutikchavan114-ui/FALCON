from __future__ import annotations

import numpy as np


def empirical_threshold(normal_scores, percentile=95.0):
    return float(np.percentile(np.asarray(normal_scores, dtype=float), percentile))


def classify(score, threshold):
    return "ANOMALOUS" if score > threshold else "NORMAL"
