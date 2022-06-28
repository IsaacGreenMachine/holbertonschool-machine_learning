#!/usr/bin/env python3
"""module for the initialize function"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:

    X - numpy.ndarray of shape (n, d)
        containing the dataset that will be used for K-means clustering
        n - number of data points
        d is the number of dimensions for each data point
    k - positive integer containing the number of clusters

    centroids are initialized with a multivariate uniform
    distribution along each dimension in d

    The min/max values for the distribution
    are min/max values of X along each dimension in d

    Returns: a numpy.ndarray of shape (k, d) containing
        the initialized centroids for each cluster, or None on failure
    """
    if (type(X) != np.ndarray or len(X.shape) != 2 or
            type(k) != int or k > X.shape[0]):
        return None
    clusters = np.random.uniform(X.min(axis=0), X.max(axis=0), (k, X.shape[1]))
    return clusters
