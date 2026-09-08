import numpy as np
from falcon.baseline import ReferenceBaseline
from falcon.anomaly import classify


def test_mahalanobis():
    X = np.array([
        [5.0, 1.00], [5.1, 1.02], [4.9, 0.98], [5.05, 1.01],
        [4.95, 1.00], [5.02, 0.99], [4.98, 1.03], [5.01, 0.97],
    ])
    baseline = ReferenceBaseline(X)
    near = baseline.score(np.array([5.0, 1.0]))
    far = baseline.score(np.array([7.0, 2.0]))
    assert far > near
    assert classify(near, 5.0) == "NORMAL"
    assert classify(far, 5.0) == "ANOMALOUS"
