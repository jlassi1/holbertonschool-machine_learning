#!/usr/bin/env python3
""" Sum squared """


def summation_i_squared(n):
    """ calculate given sum """
    if type(n) != int or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
