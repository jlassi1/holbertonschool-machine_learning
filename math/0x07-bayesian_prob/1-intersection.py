#!/usr/bin/env python3
"""1. Intersection  """
import numpy as np


def likelihood(x, n, P):
    """function that calculates the likelihood of obtaining
    this data given various hypothetical probabilities
    of developing severe side effects"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if (P > 1).any() or (P < 0).any():
        raise ValueError('All values in P must be in the range [0, 1]')
    fact = np.math.factorial(n) / (np.math.factorial(x)
                                   * np.math.factorial(n - x))
    likhd = fact * (P ** x) * ((1 - P) ** (n - x))

    return likhd


def intersection(x, n, P, Pr):
    """function that calculates the intersection of obtaining
    this data with the various hypothetical probabilities"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if (P > 1).any() or (P < 0).any():
        raise ValueError('All values in P must be in the range [0, 1]')
    if (Pr > 1).any() or (Pr < 0).any():
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), [1]):
        raise ValueError("Pr must sum to 1")
    return (likelihood(x, n, P) * Pr)
