#!/usr/bin/env python3
"""module for pca v1"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset:

    X - numpy.ndarray of shape (n, d) where:
      n is the number of data points
      d is the number of dimensions in each point
      all dimensions have a mean of 0 across all data points

    var - fraction of the variance that the PCA transformation should maintain

    Returns: the weights matrix, W, that maintains
             var fraction of X‘s original variance
    W - numpy.ndarray of shape (d, nd) where nd
        is the new dimensionality of the transformed X
    """
    u, sig, v = np.linalg.svd(X)
    sum = np.sum(sig)
    trunc = X.shape[1]
    cmsm = np.cumsum(sig)/sum
    trunc = np.where(cmsm >= var)[0][0] + 1
    return v.T[:, :trunc]
