#!/usr/bin/env python3
"""0. Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """function that determines the probability of a markov chain
    being in a particular state after a specified number of iterations"""
    state = s
    try:
        for x in range(t):
            state = np.dot(state, P)
        return state
    except Exception:
        return None
