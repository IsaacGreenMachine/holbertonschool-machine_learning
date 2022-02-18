#!/usr/bin/env python3
"""module for exponential class"""


class Exponential():
    """implements a class exponential distribution capable of stat functions"""
    def __init__(self, data=None, lambtha=1.):
        """sets lambtha based on data"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """returns PDF for exponential data"""
        if x < 0:
            return 0
        else:
            eul = 2.7182818285
            lmb = self.lambtha
            return (lmb * (eul ** (-1 * lmb * x)))

    def cdf(self, x):
        """returns CDF for poisson data"""
        if x < 0:
            return 0
        else:
            eul = 2.7182818285
            lmb = self.lambtha
            return (1 - (eul ** (-1 * lmb * x)))
