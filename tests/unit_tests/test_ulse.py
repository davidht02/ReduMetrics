import numpy as np
from ReduMetrics.metrics.ulse import ulse_score

def test_ulse_returns_float_in_0_1():
    rng = np.random.default_rng(0)
    X_high = rng.normal(size=(10, 4))
    X_low = (X_high @ rng.normal(size=(4, 2))) + 0.01 * rng.normal(size=(10, 2))
    score = ulse_score(X_high, X_low, k=3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_ulse_is_deterministic():
    rng = np.random.default_rng(1)
    X_high = rng.normal(size=(12, 5))
    X_low = X_high @ rng.normal(size=(5, 2))
    s1 = ulse_score(X_high, X_low, k=2)
    s2 = ulse_score(X_high, X_low, k=2)
    assert s1 == s2

def test_ulse_accepts_boundary_k_values():
    rng = np.random.default_rng(2)
    m, n, r = 15, 6, 2
    X_high = rng.normal(size=(m, n))
    X_low = X_high @ rng.normal(size=(n, r))
    s_k1 = ulse_score(X_high, X_low, k=1)
    s_kmax = ulse_score(X_high, X_low, k=m-1)
    assert 0.0 <= s_k1 <= 1.0
    assert 0.0 <= s_kmax <= 1.0
