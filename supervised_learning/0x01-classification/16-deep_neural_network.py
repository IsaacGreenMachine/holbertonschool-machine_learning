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
