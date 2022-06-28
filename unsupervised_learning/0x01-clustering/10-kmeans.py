#!/usr/bin/env python3
"""module for kmeans function"""
import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
    C - numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
    clss - numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to
    """
    centroid, label, inertia = sklearn.cluster.k_means(X, k)
    return centroid, label
