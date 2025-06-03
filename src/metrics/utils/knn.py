import numpy as np
from sklearn.neighbors import KDTree, BallTree

class KNNFinder:
    """
    Class to find the k-nearest neighbors using KDTree.
    """

    def __init__(self, data: np.ndarray, leaf_size: int = 40, tree_type: str = 'kd_tree'):
        """
        Initializes the KDTree with the provided data.

        Parameters
        ----------
        data : np.ndarray
            Training data with shape (n_samples, n_features).
        leaf_size : int, optional
            Number of points in the leaf nodes. Default is 40.
        """
        self.data = data
        self.tree_type = tree_type.lower()
        if self.tree_type == 'kd_tree':
            self.tree = KDTree(data, leaf_size=leaf_size)
        elif self.tree_type == 'ball_tree':
            self.tree = BallTree(data, leaf_size=leaf_size)
        else:
            raise ValueError("tree_type debe ser 'kd_tree' o 'ball_tree'")

    def query(self, points: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Finds the k-nearest neighbors for each point in 'points'.

        Parameters
        ----------
        points : np.ndarray
            Query points with shape (n_queries, n_features).
        k : int, optional
            Number of neighbors to find. Default is 5.

        Returns
        -------
        indices : np.ndarray
            Indices of the nearest neighbors in 'data' for each query point.
        """
        distances, indices = self.tree.query(points, k=k)
        return indices
