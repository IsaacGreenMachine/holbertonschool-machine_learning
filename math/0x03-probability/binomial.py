#!/usr/bin/env python3
"""module for binomial class"""


class Binomial():
    """implements a class binomial distribution capable of stat functions"""
    '''
    def __init__(self, data=None, n=1, p=0.5):
        """sets n and p based off of mean and variance of data"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = int(n)
            if p < 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                total = sum(data)
                leng = len(data)
                mean = total/leng  # = np
                var = sum([(i - mean)**2 for i in data])/leng  # = npq
                q_est = var / mean
                p_est = 1 - q_est
                n_est = (total/p_est) / leng  # " total is p% of n " formula
                self.n = int(round(n_est))
                self.p = float(mean/self.n)
    '''

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial init"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total = sum(data)
            leng = len(data)
            mean = total/leng  # = np
            var = sum([(i - mean)**2 for i in data])/leng  # = npq
            q_est = var / mean
            p_est = 1 - q_est
            n_est = (total/p_est) / leng  # " total is p% of n " formula
            self.n = int(round(n_est))
            self.p = float(mean/self.n)
    '''
            nns = []
            mean = sum(data) / len(data)
            v = sum([(i - mean) ** 2 for i in data]) / len(data)
            q = v / mean
            p = 1 - q
            self.n = round(mean / p)
            self.p = mean / self.n
    '''

    def pmf(self, k):
        """returns pmf of binomial dist"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        n = self.n
        p = self.p
        co = self.ft(n)/(self.ft(k) * self.ft(n - k))
        return co * (p ** k) * ((1 - p) ** (n - k))

    def cdf(self, k):
        """returns cdf of binomial dist"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.pmf(x) for x in range(k + 1)])

    def ft(self, n):
        """returns factorial"""
        fact = 1
        for num in range(1, n + 1):
            fact *= num
        return fact
