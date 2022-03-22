#!/usr/bin/env python3
"""module for normalization constants"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """creates train ope for a nnusing RMSProp optimization"""
    rms_op = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms_op.minimize(loss)
