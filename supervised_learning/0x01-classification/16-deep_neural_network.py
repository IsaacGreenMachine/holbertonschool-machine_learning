"""Module containing the class DeepNeuralNetwork which defines a deep neural
network performing binary classification."""
import numpy as np


class DeepNeuralNetwork():
    """Class that defines a deep neural network performing binary
    classification."""

    def __init__(self, nx, layers):
        """Class constructor
        Args:
            nx (int): The number of input features
            layers (list[int]): List representing the number of nodes in each
                layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        weights = {}
        prev = nx
        for i, l in enumerate(layers, 1):
            if not isinstance(l, int) or l < 1:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(i)] = np.zeros((l, 1))
            weights["W{}".format(i)] = (np.random.randn(l, prev) *
                                        np.sqrt(2 / prev))
            prev = l
        self.weights = weights


'''
#!/usr/bin/env python3
"""module for deep neural network class"""
import numpy as np


class DeepNeuralNetwork:
    """implements a deep nerual network with many layers"""
    def __init__(self, nx, layers):
        """sets attributes for deep neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not all(type(layer) is int and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        layers.insert(0, nx)
        self.cache = {}
        self.weights = {}
        for layer in range(1, self.L+1):
            he = np.sqrt(2/(layers[layer - 1]))
            rnd_val = np.random.randn(layers[layer], layers[layer - 1]) * he
            self.weights["W{}".format(layer)] = rnd_val
            self.weights["b{}".format(layer)] = np.zeros((layers[layer], 1))
'''
