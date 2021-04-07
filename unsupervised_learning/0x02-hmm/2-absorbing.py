#!/usr/bin/env python3
""" 2. Absorbing Chains  """
import numpy as np


def absorbing(P):
    """function that determines if a markov chain is absorbing"""
    try:
        n = P.shape[0]
        diag = np.where(np.diag(P) == 1, 1, 0)
        if not np.any(diag == 1):
            return False

        d, v = np.linalg.eig(P)
        P_bar = v @ np.diag(d == 1).astype(int) @ np.linalg.inv(v)

        if int(np.ceil(np.max(P_bar.sum()))) != n:
            return False
        return True

    except Exception:
        return False
