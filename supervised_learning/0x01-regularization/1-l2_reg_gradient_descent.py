#!/usr/bin/env python3
"""module for l2_reg_gradient_descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates weights and biases of nn using gradient descent with L2 reg

    Y - one-hot np.ndarray (classes, m) of correct labels
    classes - number of classes
    m - number of data points
    weights (dict) - weights and biases of nn
    cache (dict) - (A's) outputs of each layer of the neural network
    alpha - learning rate
    lambtha - L2 regularization parameter
    L - number of layers of the network
    nn uses tanh activations on all layers, softmax on last
    The weights and biases of the network are updated in place
    """
    m = Y.shape[1]
    # from last layer to first
    for layer in range(L, 0, -1):
        a = cache["A{}".format(layer)]
        if layer == L:
            dz = (cache["A{}".format(layer)] - Y)
        else:
            # tanh derivative func
            dz = da * (1 - np.square(a))

        l2 = ((lambtha/m) * weights["W{}".format(layer)])
        dw = (np.matmul(dz, cache["A{}".format(layer-1)].T) / m) + l2
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(weights["W{}".format(layer)].T, dz)
        new_w = weights["W{}".format(layer)] - (alpha * dw)
        new_b = weights["b{}".format(layer)] - (alpha * db)
        weights["W{}".format(layer)] = new_w
        weights["b{}".format(layer)] = new_b
