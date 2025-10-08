import numpy as np
from ReduMetrics.metrics.k_ncp import kncp_score

def test_kncp_returns_float_in_0_1():
    rng = np.random.default_rng(3)
    m = 12
    X_high = rng.normal(size=(m, 4))
    X_low  = (X_high @ rng.normal(size=(4, 2))) + 0.01 * rng.normal(size=(m, 2))
    labels = np.array([0]*(m//2) + [1]*(m - m//2))
    score = kncp_score(X_high, X_low, labels)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_kncp_is_invariant_to_label_renaming():
    rng = np.random.default_rng(4)
    m = 18
    X_high = rng.normal(size=(m, 5))
    X_low  = X_high @ rng.normal(size=(5, 2))
    # tres clases con tamaños similares
    labels = np.array([0]*(m//3) + [1]*(m//3) + [2]*(m - 2*(m//3)))
    # renombrado de clases (permuta de ids)
    mapping = {0: 2, 1: 0, 2: 1}
    labels_perm = np.vectorize(mapping.get)(labels)
    s1 = kncp_score(X_high, X_low, labels)
    s2 = kncp_score(X_high, X_low, labels_perm)
    assert s1 == s2