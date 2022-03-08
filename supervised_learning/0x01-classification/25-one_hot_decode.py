#!/usr/bin/env python3
"""module for one hot decode function"""
import numpy as np


def one_hot_decode(one_hot):
    """decodes a one-hot vector to normal vector"""
    return one_hot.T.nonzero()[1]
