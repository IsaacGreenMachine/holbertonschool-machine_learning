#!/usr/bin/env python3
"""
Contains the class DeepNeuralNetwork.
"""
import numpy as np


class DeepNeuralNetwork():
    """
    Defines a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers):
        """
        * @nx: number of input features
            * Must be int. Must be > 0.
        * @layers: list representing # of nodes in each layer of network.
            * Must be list. Must NOT be empty.
            * 1st value represents the number of nodes in the first layer.
            * Elements must ALL be positive integers.
        Sets the public instance attributes:
            * L: # of layers in neural network.
            * cache: dict to hold all intermediary values of network.
            * weights: dict to hold weights and biases of network.
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        else:
            self.L = len(layers)
            self.cache = {}
            self.weights = {}
            prev = nx
            for i in range(len(layers)):
                if type(layers[i]) is not int or layers[i] < 1:
                    raise TypeError(
                        'layers must be a list of positive integers')
                w = np.random.randn(layers[i], prev) * np.sqrt(2/prev)
                prev = layers[i]
                self.weights['W{}'.format(i + 1)] = w
                dim = len(self.weights['W{}'.format(i + 1)])
                self.weights['b{}'.format(i + 1)] = np.zeros((dim, 1))
