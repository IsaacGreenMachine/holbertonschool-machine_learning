#!/usr/bin/env python3
"""module for lenet5"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras

    X - K.Input(m, 28, 28, 1) of input images for the network
        m - number of images

    Returns: K.Model compiled with:
        - Adam optimization (default hyperparameters)
        - accuracy metrics
    """

    # setting up he_norm weight init
    init = K.initializers.he_normal()

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                            activation='relu',
                            kernel_initializer=init)(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer=init)(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # flatten conv layer output
    flat = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    dense1 = K.layers.Dense(units=120, kernel_initializer=init,
                            activation='relu')(flat)

    # Fully connected layer with 84 nodes
    dense2 = K.layers.Dense(units=84, kernel_initializer=init,
                            activation='relu')(dense1)

    # Fully connected softmax output layer with 10 nodes
    dense3 = K.layers.Dense(units=10, kernel_initializer=init,
                            activation='softmax')(dense2)

    # create model with X as input and last fully connected layer as output
    model = K.models.Model(X, dense3)

    # adam opt
    adam = K.optimizers.Adam()

    # compile model with accuracy metrics and cross_entropy loss
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
