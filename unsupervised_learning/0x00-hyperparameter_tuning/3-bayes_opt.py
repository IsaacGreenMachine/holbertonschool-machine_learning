#!/usr/bin/env python3
"""module for BayesianOptimization class"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(
            self, f, X_init, Y_init, bounds, ac_samples, l=1,
            sigma_f=1, xsi=0.01, minimize=True
                ):
        """
        f is the black-box function to be optimized

        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
            t is the number of initial samples

        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
            t is the number of initial samples

        bounds is a tuple of (min, max) representing the bounds of the
        space in which to look for the optimal point

        ac_samples is the number of samples that should be analyzed
        during acquisition

        l is the length parameter for the kernel

        sigma_f is the standard deviation given to the output of the
        black-box function

        xsi is the exploration-exploitation factor for acquisition

        minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)
        """
        min, max = bounds
        self.f = f
        # Gaussian Process for X and Y inits
        self.gp = GP(X_init, Y_init, l, sigma_f)
        # acquisition samples evenly spaced between min and max
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
