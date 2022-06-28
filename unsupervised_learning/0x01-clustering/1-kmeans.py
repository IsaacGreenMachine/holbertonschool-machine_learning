#!/usr/bin/env python3
"""Module for kmeans function"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    Args:
        X: numpy.ndarray - (n, d) containing the dataset.
            n: Number of data points.
            d: Number of dimensions for each data point.
        k: Positive integer containing the number of clusters.
        iterations: Positive integer containing the maximum
          number of iterations that should be performed.
    Return: C, clss, or None, None on failure
        C: numpy.ndarray - (k, d) containing the centroid
        means for each cluster.
        clss: numpy.ndarray - (n,) containing the index of
          the cluster in C that each data point belongs to.
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    z = 0
    clusters = np.random.uniform(X.min(axis=0), X.max(axis=0), (k, X.shape[1]))
    while (z < iterations):
        dists = []

        for i in clusters:
            # prints matrix of distances between centroid and all points
            dists.append(np.linalg.norm(X - i, axis=1))

        # which cluster each point belongs to
        mins = np.argmin(dists, axis=0)

        for count, val in enumerate(clusters):
            if np.where(mins == count)[0].size == 0:
                clusters[count] = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), (1, X.shape[1]))
                # print("new cluster: ", clusters[count])
                # print(clusters)
            else:
                clusters[count] = np.mean(X[np.where(mins == count)], axis=0)
        z += 1
    return clusters, mins
