#!/usr/bin/env python3
""" Normal Distribution"""
e = 2.7182818285
π = 3.1415926536


class Normal:
    """Normal Class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ intitializetion of class Normal"""
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                SD = 0
                for x in data:
                    SD += float((x - self.mean) ** 2)
                self.stddev = float((SD / (len(data))) ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score"""
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        coeff = (1 / (self.stddev * ((2 * π) ** 0.5)))
        pows = - 0.5 * (((x - self.mean) / self.stddev) ** 2)
        pfd = coeff * e ** pows
        return pfd

    def erf(self, x):
        """Error function"""
        a = (2 / (π ** 0.5))
        y = (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216)
        return a * y

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        y = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(y))
