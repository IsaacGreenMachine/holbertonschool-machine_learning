#!/usr/bin/env python3
"""file for np_cat function"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """returns concat of two numpy matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
