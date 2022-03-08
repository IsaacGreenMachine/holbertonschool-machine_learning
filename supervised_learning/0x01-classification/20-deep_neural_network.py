#!/usr/bin/env python3
"""module for deep neural network class"""
import numpy as np


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification."""
    def __init__(self, nx, layers):
        """implements a deep nerual network with many layers"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        elif type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        else:
            self.__L = len(layers)
            self.__cache = {}
            self.__weights = {}
            prev = nx
            for i in range(len(layers)):
                if type(layers[i]) is not int or layers[i] < 1:
                    raise TypeError(
                        'layers must be a list of positive integers')
                w = np.random.randn(layers[i], prev) * np.sqrt(2/prev)
                prev = layers[i]
                self.__weights['W{}'.format(i + 1)] = w
                dim = len(self.weights['W{}'.format(i + 1)])
                self.__weights['b{}'.format(i + 1)] = np.zeros((dim, 1))

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """single run of forward propagation for deep nn"""
        self.__cache.update({"A0": X})
        for layer in range(1, self.L+1):
            wx = self.weights["W{}".format(layer)]
            ax = self.cache["A{}".format(layer-1)]
            bx = self.weights["b{}".format(layer)]
            z = np.matmul(wx, ax) + bx
            self.__cache.update({"A{}".format(layer): 1 / (1 + np.exp(-z))})
        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """returns cost of deep nn"""
        L = -((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return (1/A.shape[1]) * np.sum(L)

    def evaluate(self, X, Y):
        """evaluates nn using forward prop and cost funcs"""
        return (self.forward_prop(X)[0].round().astype(int),
                self.cost(Y, self.cache["A{}".format(self.L)]))
