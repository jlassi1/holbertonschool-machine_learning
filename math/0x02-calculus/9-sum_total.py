#!/usr/bin/env python 3
"""Sum squared"""


def summation_i_squared(n):
    """calculate given sum"""
    if type(n) != int and n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
