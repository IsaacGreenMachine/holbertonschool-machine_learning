#!/usr/bin/env python3
"""module for gmm function"""
import sklearn.mixture


def gmm(X, k):
    """
    that calculates a GMM from a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters

    Returns: pi, m, S, clss, bic
    pi - numpy.ndarray of shape (k,) containing the cluster priors
    m - numpy.ndarray of shape (k, d) containing the centroid means
    S - numpy.ndarray of shape (k, d, d) containing the covariance matrices
    clss - numpy.ndarray of shape (n,)
        containing the cluster indices for each data point
    bic - numpy.ndarray of shape (kmax - kmin + 1)
        containing the BIC value for each cluster size tested
    """
    gm = sklearn.mixture.GaussianMixture(k).fit(X)
    return gm.weights_, gm.means_, gm.covariances_, gm.predict(X), gm.bic(X)
