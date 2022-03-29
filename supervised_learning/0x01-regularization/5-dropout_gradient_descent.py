#!/usr/bin/env python3
"""module for dropout_gradient_descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates nn weights with Dropout regularization using gradient descent:

    Y - OH np.ndarray of shape (classes, m) of correct labels for the data
    classes - number of classes
    m - number of data points
    weights - dictionary of the weights and biases of the neural network
    cache - dictionary of the outputs and dropout masks of each layer of nn
    alpha - learning rate
    keep_prob - probability that a node will be kept
    L - number of layers of the network

    All layers use tanh activation except the last (softmax activation)
    The weights of the network should be updated in place

    no return
    """
    m = Y.shape[1]
    for layer in range(L, 0, -1):
        a = cache["A{}".format(layer)]
        if layer == L:
            dz = (cache["A{}".format(layer)] - Y)
        else:
            dz = da * (1 - np.square(a))
        dw = np.matmul(dz, cache["A{}".format(layer-1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(weights["W{}".format(layer)].T, dz)

        dx = np.random.rand(da.shape[0], da.shape[1]) < keep_prob
        da *= dx
        da /= keep_prob

        new_w = weights["W{}".format(layer)] - (alpha * dw)
        new_b = weights["b{}".format(layer)] - (alpha * db)
        weights["W{}".format(layer)] = new_w
        weights["b{}".format(layer)] = new_b
