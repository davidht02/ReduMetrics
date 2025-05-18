import numpy as np
from scipy.stats import rankdata

def spearman_correlation(X_high: np.ndarray, X_low: np.ndarray, P: int = 10000, random_state: int = None) -> float:
    """
    Compute the Spearman rank-order correlation between distances in high and low dimensions.

    Parameters
    ----------
    X_high : np.ndarray, shape (m, n)
        High-dimensional data.
    X_low : np.ndarray, shape (m, r)
        Low-dimensional embedding.
    P : int, default=10000
        Number of random pairs to sample.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    rho : float
        Spearman correlation coefficient between sampled distance ranks.
    """
    m = X_high.shape[0]
    rng = np.random.default_rng(random_state)

    # Sample P pairs
    i_idx = rng.integers(0, m, size=P)
    j_idx = rng.integers(0, m, size=P)
    # ensure distinct pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    # Compute distances
    d_high = np.linalg.norm(X_high[i_idx] - X_high[j_idx], axis=1)
    d_low = np.linalg.norm(X_low[i_idx] - X_low[j_idx], axis=1)

    # Rank distances
    r_high = rankdata(d_high, method='average')
    r_low = rankdata(d_low, method='average')

    # Compute Spearman correlation
    mean_rh, mean_rl = r_high.mean(), r_low.mean()
    num = np.sum((r_high - mean_rh) * (r_low - mean_rl))
    den = np.sqrt(np.sum((r_high - mean_rh)**2) * np.sum((r_low - mean_rl)**2))
    return num / den