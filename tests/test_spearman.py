import numpy as np
from ReduMetrics.metrics.spearman import spearman_correlation

def test_spearman_returns_float_in_minus1_1():
    rng = np.random.default_rng(2)
    X_high = rng.normal(size=(30, 5))
    X_low  = (X_high @ rng.normal(size=(5, 2))) + 0.02 * rng.normal(size=(30, 2))
    rho = spearman_correlation(X_high, X_low, P=300, random_state=7)
    assert isinstance(rho, float)
    assert -1.0 <= rho <= 1.0

def test_spearman_is_reproducible_with_seed():
    rng = np.random.default_rng(5)
    X_high = rng.normal(size=(40, 6))
    X_low  = X_high @ rng.normal(size=(6, 2))
    r1 = spearman_correlation(X_high, X_low, P=200, random_state=42)
    r2 = spearman_correlation(X_high, X_low, P=200, random_state=42)
    assert r1 == r2

def test_spearman_accepts_small_P():
    rng = np.random.default_rng(6)
    X_high = rng.normal(size=(22, 4))
    X_low  = X_high @ rng.normal(size=(4, 2))
    r = spearman_correlation(X_high, X_low, P=5, random_state=1)
    assert -1.0 <= r <= 1.0