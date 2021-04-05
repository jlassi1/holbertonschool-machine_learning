#!/usr/bin/env python3
""" 1. Regular Chains """
import numpy as np


def regular(P):
    """function that determines the steady state probabilities
    of a regular markov chain"""
    if np.any(P < 0):
        return None
    if (P == np.eye(P.shape[0])).any():
        return None
    if np.linalg.det(P) == 0:
        return None
    try:
        n = P.shape[0]
        q = (P-np.eye(n))
        ones = np.ones(n)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        # print(QTQ.shape)
        bQT = np.ones((n, 1))
        # print(bQT.shape)
        return np.linalg.solve(QTQ, bQT).T
    except Exception:
        return None
