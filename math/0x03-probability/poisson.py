#!/usr/bin/env python3
""" Poisson Distribution"""
e = 2.7182818285


class Poisson:
    """Poisson Class """

    def __init__(self, data=None, lambtha=1.):
        """ intitialization of poisson class """
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
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return (e ** (- self.lambtha) * (self.lambtha ** k)) / factorial

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        pmf = []
        for i in range(k + 1):
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            pmf.append((e ** (- self.lambtha) * (self.lambtha ** i)) / fact)
        return sum(pmf)
