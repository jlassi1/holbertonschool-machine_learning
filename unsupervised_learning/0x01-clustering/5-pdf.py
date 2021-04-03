#!/usr/bin/env python3
"""5. PDF """
import numpy as np


def pdf(X, m, S):
    """function that calculates the probability density
    function of a Gaussian distribution"""
    d = X.shape[1]
    y = np.sqrt((2*np.pi)**(d)*np.linalg.det(S))
    diff = (X - m).T
    z = np.exp(-0.5*np.dot(np.dot(diff.T, np.linalg.inv(S)), diff)).diagonal()
    P = z/y
    PDF = np.where(P < 1e-300, 1e-300, P)
    return PDF
