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

    def predict(self, X_s):
        """
        returns mean and std.dev of gaussian distribution functions at
        points in X_s based on data in self.X and self.Y

        predicts the mean and standard deviation of points in
        a Gaussian process

        X_s is a numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated

        s is the number of sample points

        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean
            for each point in X_s, respectively

            sigma is a numpy.ndarray of shape (s,) containing the
            variance for each point in X_s, respectively
        """
        # kernel of X_init and X_init
        K = self.K
        # kernel of X_init and X_s
        K_s = self.kernel(self.X, X_s)
        # kernel of X_s and X_s
        K_ss = self.kernel(X_s, X_s)
        # inverse of kernel of X_init and X_init
        K_inv = np.linalg.inv(K)
        # Equation (7) from
        # http://krasserm.github.io/2018/03/19/gaussian-processes/
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        # Equation (8) from
        # http://krasserm.github.io/2018/03/19/gaussian-processes/
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        # reshaping from (x, 1) to (x, )
        mu = mu_s.reshape(-1)
        # getting diagonals of covariance matrix
        sigma = np.diag(cov_s)
        return mu, sigma
