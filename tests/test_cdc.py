import numpy as np
from ReduMetrics.metrics.cdc import cdc_score

def test_cdc_returns_float_in_minus1_1():
    rng = np.random.default_rng(4)
    m = 15
    X_high = rng.normal(size=(m, 5))
    X_low  = (X_high @ rng.normal(size=(5, 2))) + 0.02 * rng.normal(size=(m, 2))
    labels = np.array([0]*(m//3) + [1]*(m//3) + [2]*(m - 2*(m//3)))
    rho = cdc_score(X_high, X_low, labels)
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0

def test_cdc_is_invariant_to_label_renaming():
    rng = np.random.default_rng(5)
    m = 24
    X_high = rng.normal(size=(m, 6))
    X_low  = X_high @ rng.normal(size=(6, 2))
    labels = np.array([0]*(m//4) + [1]*(m//4) + [2]*(m//4) + [3]*(m - 3*(m//4)))
    # permuta ids de clase
    mapping = {0: 1, 1: 3, 2: 0, 3: 2}
    labels_perm = np.vectorize(mapping.get)(labels)
    r1 = cdc_score(X_high, X_low, labels)
    r2 = cdc_score(X_high, X_low, labels_perm)
    assert r1 == r2

def test_cdc_handles_two_classes_gracefully():
    rng = np.random.default_rng(6)
    m = 20
    X_high = rng.normal(size=(m, 4))
    X_low  = X_high @ rng.normal(size=(4, 2))
    labels = np.array([0]*(m//2) + [1]*(m - m//2))  # C=2
    rho = cdc_score(X_high, X_low, labels)
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0
