#!/usr/bin/env python3
"""0. Initialize K-means """
import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-mean"""
    n, d = X.shape
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    try:
        points = np.random.uniform(
            low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d))
    except Exception:
        return None
    return points
