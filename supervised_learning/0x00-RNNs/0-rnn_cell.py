#!/usr/bin/env python3
"""module for RNNCell class"""
import numpy as np


class RNNCell():
    """
    represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data

        h is the dimensionality of the hidden state

        o is the dimensionality of the outputs

        public attributes:
            Wh - hidden state + input data weights
            Wy - output weights
            bh - hidden state + input data biases
            by - output biases
        """
        # input sized for data input (i) + input from last recurrent node (h)
        self.Wh = np.random.randn(i + h, h)
        # output
        self.Wy = np.random.randn(h, o)

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that
        contains the data input for the cell
            m is the batch size for the data
            i is the dimensionality of the data

        h_prev is a numpy.ndarray of shape (m, h)
        containing the previous hidden state
            m is the batch size for the data
            h is the dimensionality of the hidden state

        Returns: h_next, y
          h_next is the next hidden state
          y is the output of the cell
        """
        # applying weights to data inputs
        # (only second half of Wh, first half is input from prev. RNN node)
        new_x = np.dot(x_t, self.Wh[h_prev.shape[1]:, :])
        # applying weights to hidden layer inputs
        # (only first half of Wh, second half is input from data)
        new_h = np.dot(h_prev, self.Wh[:h_prev.shape[1], :])
        # next node's state, adding bias
        h_next = np.tanh(new_x + new_h + self.bh)
        # output of the node, applying Wy (output) weights
        # adding bias to the finished product of the node
        node_output = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, node_output

    def softmax(self, x):
        """softmax activation function"""
        return np.exp(x) / np.sum(np.exp(x))
