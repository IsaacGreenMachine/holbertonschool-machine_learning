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
    """
    init = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters
    # first group
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv2 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    conv3 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)
    pl = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=1,
        padding='same'
    )(A_prev)

    # second group
    conv4 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv2)
    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv3)
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
    """

    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    # 1D
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

    conv3 = K.layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv3_1D)

    # 1D -> 5x5
    conv5_1D = K.layers.Conv2D(
        F5R,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(A_prev)

    conv5 = K.layers.Conv2D(
        F5,
        (5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(conv5_1D)

    # 3x3 pool -> 1D
    pool = K.layers.MaxPooling2D(
        (2, 2),
        strides=(1, 1),
        padding='same'
    )(A_prev)
    convPool = K.layers.Conv2D(
        FPP,
        (1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(pool)

    return K.layers.Concatenate()([conv1D, conv3, conv5, convPool])
