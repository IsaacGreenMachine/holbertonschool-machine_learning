#!/usr/bin/env python3
"""module for deep_rnn function"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN""

    rnn_cells is a list of RNNCell instances of length l
    that will be used for the forward propagation
        l is the number of layers

    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data

    h_0 is the initial hidden state, given as a
    numpy.ndarray of shape (l, m, h)
        l is the number of layers
        m is the batch size
        h is the dimensionality of the hidden state

    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    H, Y = [h_0], []
    # temp_h is updated with each forward function call
    # so [new, new, old, old, old] -> [new, new, new, old, old]
    # where "new" is hidden states from this layer (prev_h)
    # and "old" is output from previous layer (x_t)
    temp_h = h_0.copy()

    for layer in range(len(rnn_cells)):
        for timeStep in range(X.shape[0]):
            # get data from input instead of from RNN cell
            if layer == 0:
                temp_h[0], out = rnn_cells[0].forward(temp_h[0], X[timeStep])
            else:
                temp_h[layer], out = rnn_cells[layer].forward(
                    temp_h[layer], temp_h[layer - 1])
            if layer == len(rnn_cells) - 1:
                Y.append(out)

        H.append(temp_h.copy())

    return np.stack(H), np.stack(Y)
