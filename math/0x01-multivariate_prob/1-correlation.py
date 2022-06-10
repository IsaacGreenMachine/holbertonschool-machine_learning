#!/usr/bin/env python3
"""module for correlation function"""
from typing import Type
import numpy as np


def correlation(C):
    """returns correlation matrix for covariance matrix C"""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) < 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    variances = np.diag(C)
    stddevs = np.sqrt(variances)
    # divide horizontally, then vertically
    return (C / stddevs) / stddevs[:, np.newaxis]
