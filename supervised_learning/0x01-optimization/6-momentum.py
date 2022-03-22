#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates train op for a nn for grad desc with momentum"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
