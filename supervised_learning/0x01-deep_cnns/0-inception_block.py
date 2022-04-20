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

    init = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters

    # F1 separate layer
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    # F3R (1x1conv -> F3)
    conv2 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    # F3 (3x3 conv)
    conv4 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv2)

    # F5R (1x1 conv -> F5)
    conv3 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    # F3 (5x5 conv)
    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv3)

    # pooling layer -> FPP
    pl = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=1,
        padding='same'
    )(A_prev)

    # FPP (1x1 conv)
    conv6 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(pl)

    # connect time
    concat = K.layers.Concatenate()([conv1, conv4, conv5, conv6])
    return concat
