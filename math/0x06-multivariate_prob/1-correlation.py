#!/usr/bin/env python3
"""  1. Correlation """
import numpy as np


def correlation(C):
    """function that calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    D_inv = np.diag(1 / np.sqrt(np.diag(C)))
    return D_inv @ C @ D_inv
