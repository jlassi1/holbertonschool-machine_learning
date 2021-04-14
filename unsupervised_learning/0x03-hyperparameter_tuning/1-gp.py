#!/usr/bin/env python3
""" 1. Gaussian Process Prediction  """
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

    def predict(self, X_s):
        """function that predicts the mean and standard deviation
        of points in a Gaussian process"""
        #  Compute inerse kernel(X, X).
        K_inv = np.linalg.inv(self.kernel(self.X, self.X))
        # Compute kernel(X_s, X).
        Ksx = self.kernel(X_s, self.X)
        # Compute kernel(X_s, X_s).
        Kss = self.kernel(X_s, X_s)
        # Compute posterior mean k(X_star, X) (k(X, X) + Σ)⁻¹ Y.
        mu = np.matmul(np.matmul(Ksx, K_inv), self.Y).reshape((X_s.shape[0],))
        # Compute posterior covariance:
        # k(X_star, X_star) - k(X_star, X) (k(X, X) + Σ)⁻¹ k(X_star, X)ᵀ
        covariance = Kss - (np.matmul(np.matmul(Ksx, K_inv), Ksx.T))
        sigma = covariance.diagonal()
        return mu, sigma
