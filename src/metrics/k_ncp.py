import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def k_nearest_class_preservation(X_high: np.ndarray, X_low: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute k-Nearest Class Preservation (k-NCP) score.

    Parameters
    ----------
    X_high : np.ndarray, shape (m, n)
        High-dimensional data.
    X_low : np.ndarray, shape (m, r)
        Low-dimensional embedding.
    labels : np.ndarray, shape (m,)
        Class labels for each sample (integers 0..C-1).

    Returns
    -------
    kncp : float
        Average fraction of preserved neighbor classes.
    """
    classes = np.unique(labels)
    C = len(classes)
    # dynamic k
    k = (C + 2) // 4

    # compute centroids
    cent_high = np.vstack([X_high[labels == c].mean(axis=0) for c in classes])
    cent_low = np.vstack([X_low[labels == c].mean(axis=0) for c in classes])

    # pairwise distances
    Dn = pairwise_distances(cent_high)
    Dr = pairwise_distances(cent_low)

    preserved = 0.0
    for idx in range(C):
        # neighbors indices sorted by distance
        nh = np.argsort(Dn[idx])[1:k+1]
        nl = np.argsort(Dr[idx])[1:k+1]
        preserved += len(np.intersect1d(nh, nl)) / k

    return preserved / C