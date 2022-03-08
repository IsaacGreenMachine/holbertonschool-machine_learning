"""module for one hot encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """encodes a given vector to one-hot vector"""
    oh = np.zeros((classes, Y.size))
    oh[Y, np.arange(Y.size)] = 1
    return oh
