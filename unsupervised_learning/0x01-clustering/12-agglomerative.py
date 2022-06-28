#!/usr/bin/env python3
"""module for agglomerative function"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    The only imports you are allowed to use are:
    import scipy.cluster.hierarchy
    import matplotlib.pyplot as plt
    Returns: clss, a numpy.ndarray of shape (n,)
        containing the cluster indices for each data point
    """
    agg = scipy.cluster.hierarchy.linkage(X, 'ward')
    groups = scipy.cluster.hierarchy.fcluster(agg, dist, 'distance')
    print(groups)
    scipy.cluster.hierarchy.dendrogram(
        agg, color_threshold=dist, above_threshold_color='b')
    plt.show()
    plt.close()
    return groups
