import numpy as np
from src.metrics.utils.knn import KNNFinder

def test_knn_finder():
    X = np.random.rand(10, 3)
    knn = KNNFinder(X)
    indices = knn.query(X, k=3)
    assert indices.shape == (10, 3)
    for i in range(10):
        assert i in indices[i]