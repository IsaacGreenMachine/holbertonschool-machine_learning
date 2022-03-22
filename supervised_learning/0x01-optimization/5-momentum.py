#!/usr/bin/env python3
"""module for normalization constants"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates var using grad desc momentum optimization algorithm"""
    new = (v*beta1) + (grad*(1-beta1))
    return var - (new*alpha), new
