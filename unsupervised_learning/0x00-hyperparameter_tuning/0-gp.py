#!/usr/bin/env python3
"""module for GaussianProcess class"""
import numpy as np


class GaussianProcess():
    """represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
            t is the number of initial samples
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
            t is the number of initial samples

        HYPERPARAMETERS:
        l is the length parameter for the kernel.
            this controls the distance between X samples
        sigma_f is the standard deviation given to the output of
        the black-box function
            determines scale of Y values

        Sets the public instance attributes X, Y, l, and sigma_f
        corresponding to the respective constructor inputs

        Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
            kernel measures similarities of X1 and X2
            list of functions to be used in GPs
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Returns: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
