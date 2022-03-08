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
        self.W = np.random.normal(0, 1, size=(1, nx))
        self.b = 0
        self.A = 0
