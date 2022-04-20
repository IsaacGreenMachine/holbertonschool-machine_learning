#!/usr/bin/env python3
"""module for inception_block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds inception block from 'Going Deeper with Convolutions (2014)'

    A_prev - output from the previous layer
    filters - tuple or list containing F1, F3R, F3,F5R, F5, FPP, respectively:
        F1 - number of filters in the 1x1 convolution
        F3R - number of filters in 1x1 conv before the 3x3 conv
        F3 - number of filters in the 3x3 convolution
        F5R - number of filters in 1x1 conv before the 5x5 conv
        F5 - number of filters in the 5x5 convolution
        FPP - number of filters in the 1x1 convolution after the max pooling

    Returns: the concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    # first group
    conv1D = K.layers.Conv2D(
        F1,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv3_1D = K.layers.Conv2D(
        F3R,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv5_1D = K.layers.Conv2D(
        F5R,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    pool = K.layers.MaxPooling2D(
        (3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    # second group
    conv3 = K.layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv3_1D)

    conv5 = K.layers.Conv2D(
        F5,
        (5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv5_1D)

    convPool = K.layers.Conv2D(
        FPP,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(pool)

    return K.layers.Concatenate()([conv1D, conv3, conv5, convPool])
