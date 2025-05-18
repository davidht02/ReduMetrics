import numpy as np
from scipy.stats import rankdata

from sklearn.metrics import pairwise_distances

def centroid_distance_correlation(X_high: np.ndarray, X_low: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Centroid Distance Correlation (CDC) using Spearman correlation.

    Parameters
    ----------
    X_high : np.ndarray, shape (m, n)
        High-dimensional data.
    X_low : np.ndarray, shape (m, r)
        Low-dimensional embedding.
    labels : np.ndarray, shape (m,)
        Class labels for each sample.

    Returns
    -------
    rho : float
        Spearman correlation between centroid distances.
    """
    classes = np.unique(labels)
    C = len(classes)

    # compute centroids
    cent_high = np.vstack([X_high[labels == c].mean(axis=0) for c in classes])
    cent_low = np.vstack([X_low[labels == c].mean(axis=0) for c in classes])

    # pairwise centroid distances
    Dn = pairwise_distances(cent_high)
    Dr = pairwise_distances(cent_low)

    # extract upper-triangle distances
    iu = np.triu_indices(C, k=1)
    dn = Dn[iu]
    dr = Dr[iu]

    # rank
    rn = rankdata(dn, method='average')
    rr = rankdata(dr, method='average')

    # spearman
    num = np.sum((rn - rn.mean()) * (rr - rr.mean()))
    den = np.sqrt(np.sum((rn - rn.mean())**2) * np.sum((rr - rr.mean())**2))
    return num / den