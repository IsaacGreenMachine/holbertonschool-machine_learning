#!/usr/bin/env python3
"""module for transition_layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in
    "Densely Connected Convolutional Networks"

    X - output from the previous layer
    nb_filters - integer representing the number of filters in X
    compression - compression factor for the transition layer

    Your code should implement compression as used in DenseNet-C

    All weights should use he normal initialization

    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively

    Returns: The output of the transition layer
    and the number of filters within the output, respectively
    """
    init = K.initializers.he_normal()

    # batch norm
    bn = K.layers.BatchNormalization(axis=3)(X)

    # relu
    relu = K.layers.ReLU()(bn)

    # conv2d
    conv = K.layers.Conv2D(
        filters=int(nb_filters*compression),
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=init,
    )(relu)

    # avg pool
    pool = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding="valid",
    )(conv)

    return pool, int(nb_filters*compression)
