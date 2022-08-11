#!/usr/bin/env python3
"""module for bi_rnn function"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN

    bi_cell is an instance of BidirectinalCell that will
    be used for the forward propagation

    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data

    h_0 is the initial hidden state in the forward direction, given as
    a numpy.ndarray of shape (m, h)
        m is the batch size
        h is the dimensionality of the hidden state

    h_t is the initial hidden state in the backward direction,
    given as a numpy.ndarray of shape (m, h)
        m is the batch size
        h is the dimensionality of the hidden state

    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    forwards = []
    backwards = []
    H = []

    for timeStep in range(X.shape[0]):
        if timeStep == 0:
            h_next = bi_cell.forward(h_0, X[0])
            h_prev = bi_cell.backward(h_t, X[X.shape[0] - timeStep - 1])
        else:
            h_next = bi_cell.forward(h_next, X[timeStep])
            h_prev = bi_cell.backward(h_prev, X[X.shape[0] - 1])
        forwards.append(h_next)
        backwards.append(h_prev)
    # concatenating outputs from forward and
    # backward (t, m, h) -> (t, m, 2 * h)
    H = np.concatenate((np.stack(forwards), np.stack(backwards)), axis=2)
    return H, bi_cell.output(H)
