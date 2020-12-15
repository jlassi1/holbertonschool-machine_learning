#!/usr/bin/env python3
"""Exponential distribution """
e = 2.7182818285


class Exponential:
    """class that represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """ intitializetion of exponential class"""
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (float(sum(data) / len(data)))
