#!/usr/bin/env python3
"""module for GRUCell class"""
import numpy as np


class GRUCell():
    """
    represents a gated recurrent unit:
    """

    def __init__(self, i, h, o):
        """

        i is the dimensionality of the data

        h is the dimensionality of the hidden state

        o is the dimensionality of the outputs

        public attributes:
            wz - update gate weights
            bz - update gate biases
            wr - reset gate biases
            br reset gate biases
            wh - hidden state weights
            bh - hidden state biases
            wy - output weights
            by - output biases
        """

        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden state

        The output of the cell should use a softmax activation function

        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """

        # applying weights
        # (first half is for input, second half is for hidden state)
        updateGate_x = x_t @ self.Wz[h_prev.shape[1]:, :]
        updateGate_h = h_prev @ self.Wz[:h_prev.shape[1], :]
        # adding values together and bias
        updateGate_out = self.sigmoid(updateGate_x + updateGate_h + self.bz)

        resetGate_x = x_t @ self.Wr[h_prev.shape[1]:, :]
        resetGate_h = h_prev @ self.Wr[:h_prev.shape[1], :]
        resetGate_out = self.sigmoid(resetGate_x + resetGate_h + self.br)

        hidden_x = x_t @ self.Wh[h_prev.shape[1]:, :]
        hidden_h = (h_prev * resetGate_out) @ self.Wh[:h_prev.shape[1], :]
        hidden_out = np.tanh(hidden_x + hidden_h + self.bh)

        # output at top right of cell
        h_next = (h_prev * (1 - updateGate_out)) + \
            (updateGate_out * hidden_out)

        # applying weight and bias to output
        output = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, output

    def softmax(self, x):
        """softmax activation function"""
        # return np.exp(x) / np.sum(np.exp(x))
        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x

    def sigmoid(self, x):
        """sigmoid activation function"""
        return 1/(1 + np.exp(-x))
