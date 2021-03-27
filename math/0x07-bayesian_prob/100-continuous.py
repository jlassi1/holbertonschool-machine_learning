#!/usr/bin/env python3
""" 4. Continuous Posterior  """
from scipy import special


def posterior(x, n, p1, p2):
    """function that calculates the posterior probability
    that the probability of developing severe side effects
    falls within a specific range given the data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or (p1 > 1) or (p1 < 0):
        raise ValueError('p1 must be a float in the range [0, 1]')
    if not isinstance(p2, float) or (p2 > 1) or (p2 < 0):
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')

    P1 = special.btdtr(x+1, n-x+1, p1)
    P2 = special.btdtr(x+1, n-x+1, p2)

    return P2-P1
