#!/usr/bin/env python3
"""module for regular function"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n)
        representing the transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady
        state probabilities, or None on failure
    """
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    # We have to transpose so that Markov transitions
    #   correspond to right multiplying by a column vector.
    # np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(P.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    # Since np.isclose will return an array, we've indexed with an array
    # so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    if evec1.shape[1] == 0:
        return None
    evec1 = evec1[:, 0]
    # eigs finds complex eigenvalues and eigenvectors,
    # so you'll want the real part.
    stationary = (evec1 / evec1.sum()).real
    if np.count_nonzero(stationary <= 0) > 0:
        return None
    else:
        return stationary
