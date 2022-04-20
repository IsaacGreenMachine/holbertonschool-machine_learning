#!/usr/bin/env python3
"""module for dense_block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in
    "Densely Connected Convolutional Networks"

    X - output from the previous layer
    nb_filters - integer representing the number of filters in X
    growth_rate - growth rate for the dense block
    layers - number of layers in the dense block

    You should use the bottleneck layers used for DenseNet-B

    All weights should use he normal initialization

    All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively

    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    cat = X
    for layer in range(layers):
        # batch norm
        bn1 = K.layers.BatchNormalization(axis=3)(cat)
        # relu
        relu1 = K.layers.ReLU()(bn1)
        # conv (1x1) for bottleneck
        conv1 = K.layers.Conv2D(
            filters=growth_rate * 4,
            kernel_size=1,
            padding="same",
            kernel_initializer=init
        )(relu1)
        # batch norm
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        # relu
        relu2 = K.layers.ReLU()(bn2)
        # conv (1x1) for bottleneck
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding="same",
            kernel_initializer=init
        )(relu2)
        cat = K.layers.concatenate([cat, conv2])
        nb_filters += growth_rate

    return cat, nb_filters
