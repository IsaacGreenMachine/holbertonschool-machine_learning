#!/usr/bin/env python3
"""module for normalization constants"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creates the train op for nn using the Adam optimization"""
    adm_op = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                    beta2=beta2, epsilon=epsilon)
    return adm_op.minimize(loss)
