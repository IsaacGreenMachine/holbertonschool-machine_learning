#!/usr/bin/env python3
"""module for one_hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix

    labels:
    classes:
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
