#!/usr/bin/env python3
""" 2. Initialize """
import numpy as np


class MultiNormal:
    """class  that represents a Multivariate Normal distribution """

    def __init__(self, data):
        """ initialization"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[0] < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = data.mean(axis=1, keepdims=True)
        Xi = data - self.mean
        self.cov = np.dot(Xi, Xi.T) / (data.shape[1] - 1)
