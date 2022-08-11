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
        calculates the hidden state in the forward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden states

        Returns: the next hidden state
        """
        forward_x = x_t @ self.Whf[h_prev.shape[1]:, :]
        forward_h = h_prev @ self.Whf[:h_prev.shape[1], :]
        return np.tanh(forward_x + forward_h + self.bhf)

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction
        for one time step

        h_next is a numpy.ndarray of shape (m, h) containing
        the next hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden states

        x_t is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        Returns: h_pev, the previous hidden state
        """
        backward_x = x_t @ self.Whb[h_next.shape[1]:, :]
        backward_h = h_next @ self.Whb[:h_next.shape[1], :]
        return np.tanh(backward_x + backward_h + self.bhb)

    def output(self, H):
        """
        calculates all outputs for the RNN:

        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions,
        excluding their initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states

        Returns: the outputs
        """
        outputs = []
        for timeStep in H:
            outputs.append(self.softmax((timeStep @ self.Wy) + self.by))
        return(np.stack(outputs))

    def softmax(self, x):
        """softmax activation function"""
        # return np.exp(x) / np.sum(np.exp(x))
        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x
