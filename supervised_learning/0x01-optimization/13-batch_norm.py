#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of nn using batch norm"""
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    z_norm = (Z - mean) / ((var + epsilon) ** (1/2))
    z_norm_param = (z_norm * gamma) + beta
    return z_norm_param
