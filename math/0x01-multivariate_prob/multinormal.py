#!/usr/bin/env python3
"""module for MultiNormal class"""
import numpy as np


class MultiNormal():
    def __init__(self, data):
        """initializes mean and covariance matrix of class"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        mean, cov = self.mean_cov(data.T)
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """returns probability of x in multivariate normal distribution"""
        D = self.mean.shape[0]
        Px = (2*np.pi)**(D/2)
        Px = 1 / (Px * (np.linalg.det(self.cov)**(1/2)))
        covI = np.linalg.inv(self.cov)
        x_mu = x - self.mean
        dot = np.dot(np.dot(x_mu.T, covI), x_mu)
        return float(Px*np.exp((-1/2)*dot))

    def mean_cov(self, X):
        """returns mean and covariance matrix of matrix X"""
        if type(X) is not np.ndarray or len(X.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")
        if X.shape[0] < 2:
            raise ValueError("X must contain multiple data points")
        mean = (np.mean(X, axis=0).reshape(1, -1))
        stddev = X - mean
        cov = np.matmul(stddev.T, stddev)/(X.shape[0] - 1)
        return mean, cov
