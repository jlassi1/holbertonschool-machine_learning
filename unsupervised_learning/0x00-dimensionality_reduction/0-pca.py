#!/usr/bin/env python3
"""0. PCA """
import numpy as np


def pca(X, var=0.95):
    """function that performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X)
    v = np.cumsum(s) / np.sum(s)
    n = np.where(v <= var, 1, 0)
    n = np.sum(n)
    return vh.T[:, :n + 1]
