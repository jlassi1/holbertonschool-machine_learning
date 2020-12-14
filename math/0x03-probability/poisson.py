#!/usr/bin/env python3
""" Initialize Poisson"""

class Poisson:
    """Poisson class """
    def __init__(self, data=None, lambtha=1.):
        """ intitializetion of poisson class """
        self.data = data
        self.lambtha = lambtha
        if self.data == None:
            if self.lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            return lambtha
        else:
            if not isinstance(self.data , list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = sum(self.data) / len(self.data)
                return self.lambtha
