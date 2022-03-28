#!/usr/bin/env python3
"""module for l2_reg_create_layer"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer that includes L2 regularization:

    prev - tensor containing output of previous layer
    n - the number of nodes the new layer should contain
    activation - activation function that should be used on the layer
    lambtha - L2 regularization parameter

    Returns - output of the new layer
    """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_lay = tf.layers.Dense(n,
                             activation=activation,
                             kernel_regularizer=reg,
                             kernel_initializer=init)
