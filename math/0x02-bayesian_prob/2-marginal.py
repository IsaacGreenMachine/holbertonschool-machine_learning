#!/usr/bin/env python3
""" module for the marginal function"""
import numpy as np


def marginal(x, n, P, Pr):
    return np.sum(intersection(x, n, P, Pr))


def intersection(x, n, P, Pr):
    """
    calculates the intersection of obtaining this
    data with the various hypothetical probabilities

    x - number of patients that develop severe side effects
    n - total number of patients observed
    P - 1D numpy.ndarray containing the various hypothetical probabilities
        of developing severe side effects
        (probability of success in trial)
    Pr - 1D numpy.ndarray containing the prior beliefs of P
         (probability that it is this item in P
          (dice 1/6, 1/6, 1/6, 1/6, 1/6, 1/6))

    Returns - 1D numpy.ndarray containing the intersection
              of obtaining x and n with each probability in P, respectively
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.count_nonzero((P < 0) | (P > 1)) > 0:
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.count_nonzero((Pr < 0) | (Pr > 1)) > 0:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if False in np.isclose([Pr.sum()], [1]):
        raise ValueError("Pr must sum to 1")
    return(Pr * likelihood(x, n, P))


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not (all(P <= 1) and all(P >= 0)):
        raise ValueError("All values in P must be in the range [0, 1]")
    likely = nCr(n, x) * (P ** x) * ((1 - P) ** (n - x))
    return likely


def nCr(n, r):
    """returns the number of combinations with n choices and r possibilities"""
    return (
        np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n - r))
    )
