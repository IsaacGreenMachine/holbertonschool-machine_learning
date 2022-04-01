#!/usr/bin/env python3
"""module for train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent:

    network - model to train
    data - numpy.ndarray of shape (m, nx) containing the input data
    labels - one-hot ndarray of labels with shape (m, classes)
    batch_size - size of the batch used for mini-batch gradient descent
    epochs - number of passes through data for mini-batch gradient descent
    verbose - boolean. determines if output should be printed during training
    shuffle - boolean. if shuffle the batches every epoch.
        Normally, it is a good idea to shuffle,
        but for reproducibility, choose to set the default to False.
    validation_data - data to validate the model with, if not None
    early_stopping - boolean. indicates whether early stopping should be used
        early stopping should only be performed if validation_data exists
        early stopping should be based on validation loss
    patience - patience used for early stopping
    learning_rate_decay - boolean. if use learning rate decay
        learning rate decay should only be performed if validation_data exists
        the decay should be performed using inverse time decay
        the learning rate should decay in a stepwise fashion after each epoch
        each time the learning rate updates, Keras should print a message
    alpha - initial learning rate
    decay_rate - decay rate
    Returns: the History object generated after training the model
    """
    earlystop = early_stopping
    if earlystop and validation_data:
        earlystop = K.callbacks.EarlyStopping(
            patience=patience, verbose=verbose)
    decay = learning_rate_decay
    if decay and validation_data:
        def sched(epoch):
            """Learning rate scheduler"""
            return alpha / (1+epoch*decay_rate)
        decay = K.callbacks.LearningRateScheduler(sched, verbose=verbose)
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=[earlystop, decay]
    )
