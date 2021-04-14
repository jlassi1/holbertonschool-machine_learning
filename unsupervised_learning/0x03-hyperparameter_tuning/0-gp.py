#!/usr/bin/env python3
""" 0. Initialize Gaussian Process """
import numpy as np


class GaussianProcess:
    """class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """initialization"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """function that calculates the covariance kernel
        matrix between two matrices"""
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = np.exp(-0.5 / self.l**2 * (x1 - x2)**2)
        return self.sigma_f**2 * K
