#!/usr/bin/env python3
"""module for kmeans function"""
import sklearn.cluster as skcls


def kmeans(X, k):
    """
    performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters

    Returns: C, clss
    C - numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
    clss - numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to
    """
    centroid, label, inertia = skcls.k_means(X, k)
    return centroid, label
