#!/usr/bin/env python3
""" 1. PCA v2  """
import numpy as np


def pca(X, ndim):
    """ Function that performs PCA on a dataset: """
    X = X - np.mean(X, axis=0)
    W = np.linalg.svd(X)[2][:ndim, :].T
    return np.matmul(X, W)
