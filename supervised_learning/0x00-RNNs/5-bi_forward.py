#!/usr/bin/env python3
"""module for BidirectionalCell class"""
import numpy as np


class BidirectionalCell():
    """represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data

        h is the dimensionality of the hidden states

        o is the dimensionality of the outputs

        public attributes:
            Whf - hidden state forward weights
            bhf - hidden state forward biases
            Whb - hidden state backward weights
            bhb - hidden state backward biases
            Wy - output weights
            by - output biases
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        # now each output is the combination (*2)
        # of forward and backward directions.
        self.Wy = np.random.randn(h * 2, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the
        forward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden states

        Returns: the next hidden state
        """
        forward_x = x_t @ self.Whf[h_prev.shape[1]:, :]
        forward_h = h_prev @ self.Whf[:h_prev.shape[1], :]
        return np.tanh(forward_x + forward_h + self.bhf)
