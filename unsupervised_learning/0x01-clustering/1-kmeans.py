#!/usr/bin/env python3
"""1. K-means  """
import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-mean"""
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        n, d = X.shape
        points = np.random.uniform(
            low=np.min(X, axis=0), high=np.max(X, axis=0), size=(k, d))
    except Exception:
        return None
    return points

    return Centroids, clss


def dist(data, centers):
    """function that calcules the distant between centre and data"""
    distance = np.linalg.norm(
        (np.array(centers) - data[:, None, :]), axis=2)
    return distance


def kmeans(X, k, iterations=1000):
    """function that performs K-means on a dataset"""
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    closest_idx = np.argmin(dist(X, centroids), axis=1)
    for i in range(iterations):
        cp = centroids.copy()
        for j in range(k):
            if X[np.where(closest_idx == j)].size == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = X[np.where(closest_idx == j)].mean(axis=0)
        closest_idx = np.argmin(dist(X, centroids), axis=1)
        if np.array_equal(cp, centroids):
            break

    return centroids, closest_idx
