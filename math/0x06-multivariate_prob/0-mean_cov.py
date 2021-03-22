#!/usr/bin/env python3
"""  0. Mean and Covariance  """
import numpy as np


def mean_cov(X):
    """function that calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    # the number of data points
    N = X.shape[0]
    # comput the mean of matrix X
    mean = X.mean(axis=0, keepdims=True)

    # make a mean matrix the same shape as data for subtraction
    mean_mat = np.outer(np.ones((N, 1)), mean)
    # ( Xi - X )
    cov = X - mean_mat
    # Σ x2 x1 / N - 1
    cov = np.dot(cov.T, cov) / (N - 1)

    return mean, cov
