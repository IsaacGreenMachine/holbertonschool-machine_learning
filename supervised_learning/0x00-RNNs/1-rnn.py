#!/usr/bin/env python3
"""module for rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN

    rnn_cell is an instance of RNNCell that
    will be used for the forward propagation

    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data

    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        m is the batch size
        h is the dimensionality of the hidden state

    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """

    hiddenStates = []
    outputs = []
    h_out = h_0
    hiddenStates.append(h_0)
    for data in X:
        h_out, node_output = rnn_cell.forward(h_out, data)
        hiddenStates.append(h_out)
        outputs.append(node_output)
    return np.array(hiddenStates), np.array(outputs)
