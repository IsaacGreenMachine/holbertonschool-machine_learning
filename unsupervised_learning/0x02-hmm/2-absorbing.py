#!/usr/bin/env python3
"""module for absorbing function"""
import numpy as np


def absorbing(P):
    """
    that determines if a markov chain is absorbing:
    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
    P[i, j] is the probability of transitioning from state i to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if type(P) != np.ndarray or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    # makes sure all probabilities for a row sum to 100%
    if not np.all(P.sum(axis=1)):
        return False
    # doesn't mess with original array
    Pcp = P.copy()

    while True:
        # 1d diagonal matrix of rows that are absorbing rows
        absorbing_rows = (np.identity(P.shape[0]) == Pcp).diagonal()
        # eliminate matrices without absorbing rows
        if not np.count_nonzero(absorbing_rows):
            return False
        else:
            # all absorbing rows indices
            abs_rows = np.argwhere(absorbing_rows != 0)
            # get values of columns that have absorbing rows
            results = Pcp[:, abs_rows.flatten()]
            # find if any non-absorbing rows lead to absorbing rows
            useful_rows = np.where(np.logical_and(results > 0, results < 1))
            # if no non-absorbing rows lead to absorbing rows
            if useful_rows[0].size == 0:
                # and all rows are absorbing rows
                if np.all(absorbing_rows):
                    return True
                # otherwise
                else:
                    return False
            # set non-absorbing rows that lead to absorbing
            # rows to be absorbing rows (it works ¯\_(ツ)_/¯)
            Pcp[useful_rows[0], :] = 0
            Pcp[useful_rows[0], useful_rows[0]] = 1
