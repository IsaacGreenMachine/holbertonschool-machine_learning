#!/usr/bin/env python3
"""module for poisson class"""


class Poisson():
    """implements a class Poisson distribution capable of stat functions"""
    def __init__(self, data=None, lambtha=1.):
        """correctly sets lambda and raises errors"""
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data)/len(data))

    def pmf(self, k):
        """returns PMF for poisson data"""
        if k <= 0:
            return 0
        else:
            e = 2.7182818285
            k = int(k)
            lmb = self.lambtha
            return (e ** (-1 * lmb))*(lmb ** (k)) / self.ft(k)

    def cdf(self, k):
        """returns CDF for poisson data"""
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            sl = list(range(k+1))
            lmb = self.lambtha
            nl = (-1 * lmb)
            sl = list(map(lambda x: ((e ** nl)*(lmb ** (x)) / self.ft(x)), sl))
            return sum(sl)

    def ft(self, n):
        """returns factorial"""
        fact = 1
        for num in range(2, n + 1):
            fact *= num
        return fact
