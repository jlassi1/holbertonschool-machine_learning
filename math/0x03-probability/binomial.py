#!/usr/bin/env python3
"""Binomail Distribution"""
e = 2.7182818285
π = 3.1415926536


class Binomial:
    """Binomail Class """
    def __init__(self, data=None, n=1, p=0.5):
        """intitialization of class Binomial"""
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            SD = 0
            for x in data:
                SD += float((x - mean) ** 2)
            sqrtstddev = float(SD / (len(data)))
            self.p = 1 - sqrtstddev / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n
