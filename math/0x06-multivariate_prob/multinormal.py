#!/usr/bin/env python3
""" 2. Initialize """
import numpy as np


class MultiNormal:
    """class  that represents a Multivariate Normal distribution """

    def __init__(self, data):
        """ initialization"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = data.mean(axis=1, keepdims=True)
        Xi = data - self.mean
        self.cov = np.dot(Xi, Xi.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """ Probability density Function """
        d = self.mean.shape[0]
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        if x.ndim != 2 or (x.shape[0] != d)or (x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        arg_mean = x - self.mean
        invsqrtdet = np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        invcov = np.linalg.inv(self.cov)
        exp = np.exp(-(np.dot(np.dot(arg_mean.T, invcov), arg_mean)) / 2)
        return (1 / invsqrtdet * exp)[0][0]
