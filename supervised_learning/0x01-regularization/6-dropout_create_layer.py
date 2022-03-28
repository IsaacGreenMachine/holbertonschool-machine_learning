#!/usr/bin/env python3
"""module for dropout_create_layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout

    prev - tensor containing the output of the previous layer
    n - number of nodes the new layer should contain
    activation - activation function that should be used on the layer
    keep_prob - probability that a node will be kept

    tf.layers.Dropout.rate is "dropout rate", which is 1 - "keep rate"

    Returns: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(1 - keep_prob)
    drop_lay = tf.layers.Dense(n, activation=activation,
                               kernel_initializer=init,
                               kernel_regularizer=drop
                               )
    return drop_lay(prev)
