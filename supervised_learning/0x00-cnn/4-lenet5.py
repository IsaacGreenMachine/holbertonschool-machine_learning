#!/usr/bin/env python3
"""module for lenet5"""
import tensorflow as tf


def lenet5(x, y):
    """builds a modified version of the LeNet-5 architecture using tensorflow

    x - tf.placeholder(m, 28, 28, 1) of input images for the network
        m - the number of images

    y - tf.placeholder(m, 10) of one-hot labels for the network
        m - the number of images

    Returns:
    a tensor for the softmax activated output
    Adam optimization training op (default hyperparameters)
    a tensor for the loss of the netowrk
    a tensor for the accuracy of the network
    """

    """
    # he_norm weight initializer
    he_norm = tf.initializers.variance_scaling()

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.Conv2D(6,
                             5,
                             padding='same',
                             activation='relu',
                             kernel_initializer=he_norm)(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.MaxPooling2D(2, 2)(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.Conv2D(16,
                             5,
                             padding='valid',
                             activation='relu',
                             kernel_initializer=he_norm)(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.MaxPooling2D(2, 2)(conv2)

    # flatten output of convolutions for densely connected layers
    pool2_flat = tf.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    dense1 = tf.layers.Dense(120,
                             activation='relu',
                             kernel_initializer=he_norm)(pool2_flat)

    # Fully connected layer with 84 nodes
    dense2 = tf.layers.Dense(84,
                             activation='relu',
                             kernel_initializer=he_norm)(dense1)

    # Fully connected softmax output layer with 10 nodes
    dense3 = tf.layers.Dense(10,
                             activation='softmax',
                             kernel_initializer=he_norm)(dense2)

    # loss
    loss = tf.losses.softmax_cross_entropy(y, dense3)

    # adam optimizer
    opt = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    y_out = tf.argmax(y, axis=1)
    pred_out = tf.argmax(dense3, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.math.equal(y_out, pred_out), tf.float32))

    return dense3, opt.minimize(loss), loss, acc
    """

    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
        filters=6, kernel_size=5, padding='same',
        activation='relu', kernel_initializer=init
    )(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=(2, 2))(conv1)

    conv2 = tf.layers.Conv2D(
        filters=16, kernel_size=5, padding='valid',
        activation='relu', kernel_initializer=init
    )(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    flat = tf.layers.Flatten()(pool2)

    fc1 = tf.layers.Dense(
        units=120, kernel_initializer=init, activation='relu'
        )(flat)

    fc2 = tf.layers.Dense(
        units=84, kernel_initializer=init, activation='relu'
        )(fc1)

    fc3 = tf.layers.Dense(units=10, kernel_initializer=init)(fc2)
    prediction = fc3

    loss = tf.losses.softmax_cross_entropy(y, fc3)
    Adam = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    softmax = tf.nn.softmax(prediction)
    return softmax, Adam, loss, accuracy
