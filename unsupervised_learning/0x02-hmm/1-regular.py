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
    try:
        n = P.shape[0]
        q = (P-np.eye(n))
        ones = np.ones(n)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(n)
        return np.linalg.solve(QTQ, bQT)
    except Exception:
        return None
