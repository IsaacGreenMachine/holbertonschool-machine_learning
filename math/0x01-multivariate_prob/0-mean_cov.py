#!/usr/bin/env python3
"""module for mean_cov function"""
import numpy as np


def mean_cov(X):
    """returns mean and covariance matrix of matrix X"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = (np.mean(X, axis=0).reshape(1, -1))
    stddev = X - mean
    cov = np.matmul(stddev.T, stddev)/(X.shape[0] - 1)
    return mean, cov
