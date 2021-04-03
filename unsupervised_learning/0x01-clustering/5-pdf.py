#!/usr/bin/env python3
"""5. PDF """
import numpy as np


def pdf(X, m, S):
    """function that calculates the probability density
    function of a Gaussian distribution"""
    if not (isinstance(X, np.ndarray)
            or isinstance(m, np.ndarray)
            or isinstance(S, np.ndarray)):
        return None
    if X.ndim != 2:
        return None
    d = X.shape[1]
    if m.ndim != 1 or m.shape != (d,):
        return None
    if S.ndim != 2 or S.shape != (d, d):
        return None

    y = np.sqrt((2*np.pi)**(d)*np.linalg.det(S))
    diff = (X - m).T
    z = np.exp(-0.5*np.dot(np.dot(diff.T, np.linalg.inv(S)), diff)
               ).diagonal()
    P = z/y
    PDF = np.where(P < 1e-300, 1e-300, P)
    return PDF
