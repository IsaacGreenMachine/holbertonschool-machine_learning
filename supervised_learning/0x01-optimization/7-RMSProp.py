#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm"""
    sdw = (beta2*s) + ((1-beta2)*(grad**2))
    return var - (alpha * (grad/((sdw ** (1/2)) + epsilon))), sdw
