#!/usr/bin/env python3
"""module for predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    makes a prediction using a neural network

    network - network model to make the prediction with
    data - input data to make the prediction with
    verbose - boolean. determines if output should print during prediction
    Returns: the prediction for the data
    """

    return network.predict(data, verbose=verbose)
