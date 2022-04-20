#!/usr/bin/env python3
"""module for inception_network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network from 'Going Deeper with Convolutions (2014)'

    input data will have shape (224, 224, 3)

    Returns: the keras model
    """

    # input size (224 x 224 x 3)
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    # layer type - output size

    # convolution - (112 x 112 x 64)
    conv2d_0 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="same",
        activation='relu',
        kernel_initializer=init
    )(input)

    # max pool - (56 x 56 x 64)
    max_pooling2d_0 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(conv2d_0)

    # 1 x 1 reduce for convolution
    conv_2d_1_red = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(max_pooling2d_0)

    # convolution - (56 x 56 x 192)
    conv2d_1 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding="same",
        activation='relu',
        kernel_initializer=init
    )(conv_2d_1_red)

    # max pool - (28 x 28 x 192)
    max_pool_2 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
        )(conv2d_1)

    # inception (3a) - (28 x 28 x 256)
    # [F1, F3R, F3, F5R, F5, FPP]
    incep_1 = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])

    # inception (3b) - (28 x 28 x 480)
    incep_2 = inception_block(incep_1, [128, 128, 192, 32, 96, 64])

    # max pool - (14 x 14 x 480)
    max_pool_3 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
        )(incep_2)

    # inception (4a) - (14 x 14 x 512)
    incep_3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])

    # inception (4b) - (14 x 14 x 512)
    incep_4 = inception_block(incep_3, [160, 112, 224, 24, 64, 64])

    # inception (4c) - (14 x 14 x 512)
    incep_5 = inception_block(incep_4, [128, 128, 256, 24, 64, 64])

    # inception (4d) - (14 x 14 x 528)
    incep_6 = inception_block(incep_5, [112, 144, 288, 32, 64, 64])

    # inception (4e) - (14 x 14 x 832)
    incep_7 = inception_block(incep_6, [256, 160, 320, 32, 128, 128])

    # max pool - (7 x 7 x 832)
    max_pool_4 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
        )(incep_7)

    # inception (5a) - (7 x 7 x 832)
    incep_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])

    # inception (5b) - (7 x 7 x 1024)
    incep_9 = inception_block(max_pool_4, [384, 192, 384, 48, 128, 128])

    # avg pool - (1 x 1 x 1024)
    avg_pool_1 = K.layers.AveragePooling2D(
        (7, 7), strides=(7, 7), padding='same'
        )(incep_9)

    # dropout (40%) - (1 x 1 x 1024)
    dropout_1 = K.layers.Dropout(.4)(avg_pool_1)

    # dense takes care of "linear",
    # activation takes care of "softmax" - (1 x 1 x 1000)
    softmax_1 = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init,
        )(dropout_1)

    return K.Model(inputs=input, outputs=softmax_1)
