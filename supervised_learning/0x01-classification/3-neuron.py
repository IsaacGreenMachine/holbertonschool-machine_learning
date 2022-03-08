#!/usr/bin/env python3
"""Module for neuron class"""
import numpy as np


class Neuron:
    """neuron with sigmoid activation"""
    def __init__(self, nx):
        """sets values for neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """get W value"""
        return self.__W

    @property
    def b(self):
        """get b value"""
        return self.__b

    @property
    def A(self):
        """get A value"""
        return self.__A

    def forward_prop(self, X):
        """implements single run of forward propagation"""
        z = np.matmul(self.W, X) + self.b
        self.__A = (1/(1 + np.exp(-z)))
        return self.A

    def cost(self, Y, A):
        """returns cost function of network"""
        L = -(Y * np.log(A)) - (1 - Y) * np.log(1.0000001 - A)
        return (1/A.shape[1]) * np.sum(L)
