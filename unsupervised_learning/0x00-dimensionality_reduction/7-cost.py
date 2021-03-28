#!/usr/bin/env python3
""" 7. Cost """

import numpy as np


def cost(P, Q):
    """
    functuion that calculates the cost of the t-SNE transformation:

    P is a numpy.ndarray of shape (n, n) containing the P affinities
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    Returns: C, the cost of the transformation
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
