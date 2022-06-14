#!/usr/bin/env python3
""" module for the likelihood function"""
import numpy as np


def likelihood(x, n, P):
    """
    You are conducting a study on a revolutionary cancer drug
    and are looking to find the probability that a patient who
    takes this drug will develop severe side effects. During
    your trials, n patients take the drug and x patients develop
    severe side effects. You can assume that x follows a binomial distribution.

    calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects

    x - number of patients that develop severe side effects
    n - total number of patients observed
    P - 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects

    Returns - 1D numpy.ndarray containing the likelihood of obtaining the data,
              x and n, for each probability in P, respectively
    """
    if type(n) != int or n < 0:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) != np.ndarray or len(P.shape) != 1:
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
