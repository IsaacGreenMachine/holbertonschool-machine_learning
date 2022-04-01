#!/usr/bin/env python3
"""module for train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent:

    network - model to train
    data - numpy.ndarray of shape (m, nx) containing the input data
    labels - one-hot ndarray of labels with shape (m, classes)
    batch_size - size of the batch used for mini-batch gradient descent
    epochs - number of passes through data for mini-batch gradient descent
    verbose - boolean. if output should be printed during training
    shuffle - boolean. if shuffle the batches every epoch.
        Normally, it is a good idea to shuffle,
        but for reproducibility, choose to set the default to False.
    Returns: the History object generated after training the model
    """

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
    )
