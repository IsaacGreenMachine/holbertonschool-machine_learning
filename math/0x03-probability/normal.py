#!/usr/bin/env python3
"""module for normal class"""


class Normal():
    """implements a normal distribution capable of stat funs"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """sets mean and stddev from data"""
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data)/len(data))
                variance = []
                for i in data:
                    variance.append((i - self.mean) ** 2)
                variance = sum(variance)/(len(data))
                self.stddev = variance ** (1/2)

    def z_score(self, x):
        """returns z score from x value"""
        return ((x - self.mean)/self.stddev)

    def x_value(self, z):
        """returns x value from z score"""
        return ((z * self.stddev) + self.mean)

    def pdf(self, x):
        """returns pdf of normal dist"""
        eu = 2.7182818285
        pi = 3.1415926536
        st = self.stddev
        mn = self.mean
        exp = (-1 * (((x - mn) ** 2)/(2 * (st ** 2))))
        return ((eu ** exp) / (st * ((2 * pi) ** (1/2))))

    def cdf(self, x):
        """returns cdf of normal dist"""
        pi = 3.1415926536
        mn = self.mean
        std = self.stddev
        return (0.5 * (1 + self.erf((x - mn) / ((2**(1/2)) * std))))

    def erf(self, x):
        """error func used in cdf"""
        pi = 3.1415926536
        out = 2 / (pi ** (1/2))
        first = (x**3)/3
        second = (x**5)/10
        third = (x**7)/42
        fourth = (x**9)/216
        return (out * (x - first + second - third + fourth))
