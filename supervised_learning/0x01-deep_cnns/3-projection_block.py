#!/usr/bin/env python3
"""module for projection_block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    builds a projection block from
    "Deep Residual Learning for Image Recognition" (2015)

    A_prev - output from the previous layer
    filters - tuple or list containing F11, F3, F12, respectively:
        F11 - number of filters in the first 1x1 convolution
        F3 - number of filters in the 3x3 convolution
        F12 - num. filters in 2nd 1x1 conv and 1x1 convolution in shortcut
    s - stride of first conv in main path and the shortcut connection

    All convolutions inside the block should be followed by
    batch normalization along the channels axis
    and a rectified linear activation (ReLU), respectively.

    All weights should use he normal initialization

    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    # (1x1) conv
    conv_0 = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
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

    # (1x1) conv on A_prev
    adjusted_input = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        padding="same",
        kernel_initializer=init,
        )(A_prev)

    # batch norm
    bn_a = K.layers.BatchNormalization(
        axis=3
        )(adjusted_input)

    # f(x) + x
    add = K.layers.Add()([bn_2, bn_a])

    # ReLu
    re_2 = K.layers.ReLU()(add)

    return(re_2)
