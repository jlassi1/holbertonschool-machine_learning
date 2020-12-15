#!/usr/bin/env python3
""" Poisson Distribution"""


class Poisson:
    """Poisson Class """
    e = 2.7182818285
    def __init__(self, data=None, lambtha=1.):
        """ intitializetion of poisson class """
        self.data = data
        if data == None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data , list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = sum(data) / len(data)


    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if not isinstance(k, int):
            self.k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1,k + 1):
            factorial = factorial*i
        return ((self.e**(-self.lambtha) *(self.lambtha**k)) / factorial)
