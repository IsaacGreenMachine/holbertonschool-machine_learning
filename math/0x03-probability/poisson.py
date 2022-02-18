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
        """returns PMF for poisson datan"""
        e = 2.7182818285
        if k < 0:
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
        if type(k) is not int:
            k = int(k)
        else:
            e = 2.7182818285
            lst = 0
            for i in range(k + 1):
                lst += self.pmf(i)
            return lst

    def ft(self, n):
        """returns factorial"""
        fact = 1
        for num in range(1, n + 1):
            fact *= num
        return fact
