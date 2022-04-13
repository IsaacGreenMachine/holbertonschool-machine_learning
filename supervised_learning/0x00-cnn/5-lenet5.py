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

    """
    model = K.Sequential()

    # input layer
    model.add(X)

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    model.add(K.layers.Conv2D(6,
                              5,
                              padding="same",
                              activation="relu",
                              kernel_initializer='HeNormal'))

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    model.add(K.layers.MaxPooling2D(2, 2))

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    model.add(K.layers.Conv2D(16,
                              5,
                              padding="valid",
                              activation="relu",
                              kernel_initializer='HeNormal'))

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    model.add(K.layers.MaxPooling2D(2, 2))

    # flatten conv layer output
    model.add(K.layers.Flatten())

    # Fully connected layer with 120 nodes
    model.add(K.layers.Dense(120,
                             activation='relu',
                             kernel_initializer='HeNormal'))

    # Fully connected layer with 84 nodes
    model.add(K.layers.Dense(84,
                             activation='relu',
                             kernel_initializer='HeNormal'))

    # Fully connected softmax output layer with 10 nodes
    model.add(K.layers.Dense(10,
                             activation='softmax',
                             kernel_initializer='HeNormal'))

    # adam opt
    opt = K.optimizers.Adam()

    # compiling with adam opt and accuracy metrics
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
    """

    init = K.initializers.he_normal()
    c1 = K.layers.Conv2D(6, kernel_size=(5, 5), padding='same',
                         activation='relu',
                         kernel_initializer=init)(X)
    p1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = K.layers.Conv2D(16, kernel_size=(5, 5), padding='valid',
                         activation='relu',
                         kernel_initializer=init)(p1)
    p2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)
    flat = K.layers.Flatten()(p2)
    layer1 = K.layers.Dense(units=120, kernel_initializer=init,
                            activation='relu')(flat)
    layer2 = K.layers.Dense(units=84, kernel_initializer=init,
                            activation='relu')(layer1)
    layer3 = K.layers.Dense(units=10, kernel_initializer=init,
                            activation='softmax')(layer2)
    newModel = K.models.Model(X, layer3)
    adam = K.optimizers.Adam()
    newModel.compile(optimizer=adam,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return newModel
