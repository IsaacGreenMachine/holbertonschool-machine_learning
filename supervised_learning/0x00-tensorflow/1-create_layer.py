#!/usr/bin/env python3
"""module for create_layer function"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """creates a tf layer with n inputs and activation func"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.Dense(prev,
                           n,
                           activation=activation,
                           kernel_initializer=weights)
