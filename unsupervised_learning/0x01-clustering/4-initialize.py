#!/usr/bin/env python3
"""4. Initialize GMM """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function that initializes variables for a Gaussian Mixture Model"""
    # if not isinstance(k, int) or k < 1:
    #     return None, None, None
    try:
        # centroid means for each cluster, initialized with K-means
        C, clss = kmeans(X, k)
        # the priors for each cluster, initialized evenly
        pi = np.full(shape=k, fill_value=1 / k)
        # the covariance matrices for each cluster, initialized as identity
        # matrices
        n, d = X.shape
        id1 = [np.identity(d)]
        S = np.array(id1 * k)
        return pi, C, S
    except Exception:
        return None, None, None
