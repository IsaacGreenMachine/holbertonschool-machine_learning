#!/usr/bin/env python3
"""module for densenet121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    that builds the DenseNet-121 architecture as described in
    "Densely Connected Convolutional Networks"

    growth_rate - growth rate
    compression - compression factor

    You can assume the input data will have shape (224, 224, 3)

    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively

    All weights should use he normal initialization

    Returns: the keras model
    """

    """

    # relu
    relu = K.layers.ReLU()(bn)

    """

    init = K.initializers.he_normal()
    input = K.Input(shape=(224, 224, 3))

    # batch norm
    bn = K.layers.BatchNormalization(axis=3)(input)

    X = K.layers.Activation('relu')(bn)

    # 7x7 conv /2
    conv = K.layers.Conv2D(
        filters=2*growth_rate,  # page 4 under "implementation details"
        kernel_size=7,
        strides=2,
        padding="same",
        kernel_initializer=init,
    )(X)

    # 3x3 max pool /2
    pool = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
    )(conv)

    # dense 1
    d1, nb_filters = dense_block(pool, 64, growth_rate, 6)
    # transition 1
    d1_t, nb_filters = transition_layer(d1, nb_filters, compression)
    # dense 2
    d2, nb_filters = dense_block(d1_t, nb_filters, growth_rate, 12)
    # transition 2
    d2_t, nb_filters = transition_layer(d2, nb_filters, compression)
    # dense 3
    d3, nb_filters = dense_block(d2_t, nb_filters, growth_rate, 24)
    # transition 3
    d3_t, nb_filters = transition_layer(d3, nb_filters, compression)
    # dense 4
    d4, nb_filters = dense_block(d3_t, nb_filters, growth_rate, 16)

    # classification
    # 7x7 avg pool
    avg_pool = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding="valid",
    )(d4)

        # fully connected
    softmax = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init,
    )(avg_pool)

    return K.Model(inputs=input, outputs=softmax)
