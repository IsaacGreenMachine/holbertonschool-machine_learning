#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    m = np.sum(X, axis=0) / X.shape[0]
    s = np.sqrt(np.sum((X - m)**2, axis=0) / X.shape[0])
    return (m, s)
