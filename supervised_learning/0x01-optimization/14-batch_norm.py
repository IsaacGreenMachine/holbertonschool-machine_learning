#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch norm layer for a nn in tensorflow"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.dense(prev, n, kernel_initializer=weights)
    mean, var = tf.nn.moments(lay, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1/100000000
    bn_lay = tf.nn.batch_normalization(lay, mean, var, beta, gamma, epsilon)
    return activation(bn_lay)
