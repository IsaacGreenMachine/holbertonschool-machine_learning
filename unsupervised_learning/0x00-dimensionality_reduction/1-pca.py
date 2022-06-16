#!/usr/bin/env python3
"""module for pca v1"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X

    Returns: T, a numpy.ndarray of shape (n, ndim)
             containing the transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    u, sig, v = np.linalg.svd(X_m)
    return np.matmul(X_m, v.T[:, :ndim])
