#!/usr/bin/env python3
"""module for dropout_forward_prop"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout

    X - ndarray of shape (nx, m) containing the input data for the network
    nx - number of input features
    m - number of data points
    weights - dictionary of the weights and biases of nn
    L - number of layers in the network
    keep_prob - probability that a node will be kept

    All layers except the last use the tanh activation function
    The last layer uses the softmax activation function

    Returns: dictionary of layer outputs and layer dropout masks
    """
    outputs = {}
    outputs.update({"A0": X})
    for layer in range(1, L+1):
        w = weights["W{}".format(layer)]
        a = outputs["A{}".format(layer-1)]

        if layer < L:
            dx = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            outputs["D{}".format(layer)] = dx*1
            a *= dx
            a /= keep_prob

        b = weights["b{}".format(layer)]
        z = np.matmul(w, a) + b
        if layer == L:
            t = np.exp(z)
            outputs["A{}".format(layer)] = t/np.sum(t, axis=0)
        else:
            top = np.exp(z) - np.exp(-z)
            bot = np.exp(z) + np.exp(-z)
            outputs["A{}".format(layer)] = top / bot
    return outputs
