#!/usr/bin/env python3
"""module for test_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    tests a neural network

    network - network model to test
    data - input data to test the model with
    labels - correct one-hot labels of data
    verbose - boolean. determines if output should be printed during testing
    Returns: loss and accuracy of  model with the testing data, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
