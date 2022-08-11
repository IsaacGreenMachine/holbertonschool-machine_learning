#!/usr/bin/env python3
"""module for LSTMCell class"""
import numpy as np


class LSTMCell():
    """represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Public Attributes:
            Wf - forget gate weights
            Wu - update or "input" gate weights
            Wc - candidate (intermediate) state weights
            Wo - output gate weights
            Wy - output weights
            bf - forget gate biases
            bu - update or "input" gate biases
            bc - candidate (intermediate) state biases
            bo - output gate biases
            by - output biases
        """
        # forget gate
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # update gate
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # hidden state
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # output gate
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # ouptut
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step
        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden state
        c_prev is a numpy.ndarray of shape (m, h)
        containing the previous cell state
            m is the batch size for the data
            h is the dimensionality of the hidden state
        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """

        forgetGate_x = x_t @ self.Wf[h_prev.shape[1]:, :]
        forgetGate_h = h_prev @ self.Wf[:h_prev.shape[1], :]
        forgetGate_out = self.sigmoid(forgetGate_x + forgetGate_h + self.bf)
        forgetGate_c = c_prev * forgetGate_out

        updateGate_x = x_t @ self.Wu[h_prev.shape[1]:, :]
        updateGate_h = h_prev @ self.Wu[:h_prev.shape[1], :]
        updateGate_out = self.sigmoid(updateGate_x + updateGate_h + self.bu)

        intermediateGate_x = x_t @ self.Wc[h_prev.shape[1]:, :]
        intermediateGate_h = h_prev @ self.Wc[:h_prev.shape[1], :]
        intermediateGate_out = np.tanh(
            intermediateGate_x + intermediateGate_h + self.bc)
        c_next = (intermediateGate_out * updateGate_out) + forgetGate_c

        outputGate_x = x_t @ self.Wo[h_prev.shape[1]:, :]
        outputGate_h = h_prev @ self.Wo[:h_prev.shape[1], :]
        outputGate_out = self.sigmoid(outputGate_x + outputGate_h + self.bo)
        h_next = np.tanh(c_next) * outputGate_out

        output = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, output

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
