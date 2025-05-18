import numpy as np
from src.metrics.utils.knn import KNNFinder

def ulse_score(X_high: np.ndarray, X_low: np.ndarray, k: int = 5) -> float:
    """
    Compute the Unsupervised Local Structure Evaluation (ULSE) score.

    Parameters
    ----------
    X_high : array-like, shape (m, n)
        High-dimensional data (m samples, n features).
    X_low : array-like, shape (m, r)
        Low-dimensional embedding (m samples, r features).
    k : int, default=5
        Number of nearest neighbors to consider.

    Returns
    -------
    ulse : float
        The average proportion of preserved local neighbors.
    """
    m = X_high.shape[0]

    knn_high = KNNFinder(X_high)
    knn_low = KNNFinder(X_low)

    # Obtener los índices de los k vecinos más cercanos para cada punto
    neighs_high = knn_high.query(X_high, k=k)
    neighs_low = knn_low.query(X_low, k=k)

    # Compute ULSE
    preserved = 0.0
    for i in range(m):
        set_high = set(neighs_high[i])
        set_low = set(neighs_low[i])
        preserved += len(set_high.intersection(set_low)) / k

    return preserved / m