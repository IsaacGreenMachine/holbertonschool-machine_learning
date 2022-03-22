#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    print(X.shape)
    m = np.sum(X, axis=0) / X.shape[0]
    s = np.sqrt(np.sum((X - m)**2, axis=0) / X.shape[0])
    return (m, s)
