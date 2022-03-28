#!/usr/bin/env python3
"""module for l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization

    cost - cost of the network without L2 regularization
    lambtha - regularization parameter
    weights - dictionary of weights and biases (np.ndarrays) of nn
    L - number of layers in the neural network
    m - number of data points used

    Returns: the cost of the network accounting for L2 regularization
    """
    frobenius = []
    for k, w in weights.items():
        if 'W' in k:
            frobenius.append(np.sum(w**2))
    return cost + ((lambtha/(2*m)) * np.sum(frobenius))
