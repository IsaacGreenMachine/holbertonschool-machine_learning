#!/usr/bin/env python3
"""module for one hot encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """encodes a given vector to one-hot vector"""
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if classes < 2 or classes < np.amax(Y):
        return None
    oh = np.zeros((classes, Y.size))
    oh[Y, np.arange(Y.size)] = 1
    return oh
