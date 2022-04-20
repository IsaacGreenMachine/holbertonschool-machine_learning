#!/usr/bin/env python3
"""module for identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds identity block from
    "Deep Residual Learning for Image Recognition" (2015):

    A_prev - output from the previous layer
    filters - tuple or list containing F11, F3, F12, respectively:
        F11 - number of filters in the first 1x1 convolution
        F3 - number of filters in the 3x3 convolution
        F12 - number of filters in the second 1x1 convolution

    All convolutions inside the block should be followed by
    batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.

    All weights should use he_normal initialization

    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    # (1x1) conv
    conv_0 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=init,
        )(A_prev)

    # batch norm
    bn_0 = K.layers.BatchNormalization(
        axis=3
        )(conv_0)

    # ReLu
    re_0 = K.layers.ReLU()(bn_0)

    # (3x3) conv
    conv_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=init,
        )(re_0)

    # batch norm
    bn_1 = K.layers.BatchNormalization(
        axis=3
        )(conv_1)

    # ReLu
    re_1 = K.layers.ReLU()(bn_1)

    # (1x1) conv
    conv_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=init,
        )(re_1)

    # batch norm
    bn_2 = K.layers.BatchNormalization(
        axis=3
        )(conv_2)

    # f(x) + x
    add = K.layers.Add()([bn_2, A_prev])

    # ReLu
    re_2 = K.layers.ReLU()(add)

    return(re_2)
