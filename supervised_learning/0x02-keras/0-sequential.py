#!/usr/bin/env python3
"""module for build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library, Sequential Class

    nx - number of input features to the network
    layers - list containing the number of nodes in each layer of the network
    activations - list of layer activation functions
    lambtha - L2 regularization parameter
    keep_prob - probability that a node will be kept for dropout

    Returns: the keras model
    """
    model = K.Sequential()
    model.add(K.layers.InputLayer(nx,))
    for i in range(len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
            )
            )
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
