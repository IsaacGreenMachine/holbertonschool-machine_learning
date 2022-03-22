#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]
