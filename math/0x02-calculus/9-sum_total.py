#!/usr/bin/env python3
""" summation function """


def summation_i_squared(n):
    """ a functionthat calculates [\\sum_{i=1}^{n} i^2] """
    if not isinstance(n, int):
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
