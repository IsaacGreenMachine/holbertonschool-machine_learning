#!/usr/bin/env python3
"""module for markov_chain function"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a
    particular state after a specified number of iterations

    P - square 2D numpy.ndarray of shape (n, n)
        representing the transition matrix
      n is the number of states in the markov chain
      P[i, j] is the probability of transitioning from state i to state j

    s - numpy.ndarray of shape (1, n) representing
        the probability of starting in each state

    t - number of iterations that the markov chain has been through

    Returns: numpy.ndarray of shape (1, n) representing the probability
        of being in a specific state after t iterations, or None on failure
    """
    if (
        (type(P), type(s), type(t)) != (np.ndarray, np.ndarray, int) or
        s.shape[1] != P.shape[0] or
        P.shape[0] != P.shape[1]
    ):
        return None
    else:
        out_x = s
        for i in range(t):
            out_x = np.matmul(out_x, P)
        return out_x
