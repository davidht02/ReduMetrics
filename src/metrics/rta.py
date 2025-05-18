import numpy as np

def rta_score(X_high: np.ndarray, X_low: np.ndarray, T: int = 10000, random_state: int = None) -> float:
    """
    Compute Random Triplet Accuracy (RTA) for dimensionality reduction.

    Parameters
    ----------
    X_high : np.ndarray, shape (m, n)
        High-dimensional data.
    X_low : np.ndarray, shape (m, r)
        Low-dimensional embedding.
    T : int, default=10000
        Number of random triplets to sample.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    rta : float
        Fraction of triplets preserving relative order of distances.
    """
    m = X_high.shape[0]
    rng = np.random.default_rng(random_state)

    # Sample T random triplets (i, j, l)
    # Ensure i, j, l are distinct for each triplet
    triplets = rng.choice(m, size=(T, 3), replace=True)
    # filter out any with duplicates within the triplet
    mask = np.logical_and(
        triplets[:, 0] != triplets[:, 1],
        np.logical_and(triplets[:, 0] != triplets[:, 2], triplets[:, 1] != triplets[:, 2])
    )
    triplets = triplets[mask]
    # If filtering reduced below T, resample additional until length >= T
    while triplets.shape[0] < T:
        extra = rng.choice(m, size=(T, 3), replace=True)
        mask = np.logical_and(
            extra[:, 0] != extra[:, 1],
            np.logical_and(extra[:, 0] != extra[:, 2], extra[:, 1] != extra[:, 2])
        )
        triplets = np.vstack([triplets, extra[mask]])
    triplets = triplets[:T]

    # Compute distances and count preserved triplets
    preserved = 0
    for i, j, l in triplets:
        d1 = np.linalg.norm(X_high[i] - X_high[j])
        d2 = np.linalg.norm(X_high[i] - X_high[l])
        d1p = np.linalg.norm(X_low[i] - X_low[j])
        d2p = np.linalg.norm(X_low[i] - X_low[l])
        # preserve if both orders match
        if (d1 < d2 and d1p < d2p) or (d1 > d2 and d1p > d2p):
            preserved += 1

    return preserved / T