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

    # he_norm weight initializer for layers
    init = tf.contrib.layers.variance_scaling_initializer()

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(
        x, filters=6, kernel_size=5, padding='same',
        activation='relu', kernel_initializer=init
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(
        pool1, filters=16, kernel_size=5, padding='valid',
        activation='relu', kernel_initializer=init
    )

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # flattening convolution for densely connected layer
    pool2_flat = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    dense1 = tf.layers.dense(
        pool2_flat, units=120, kernel_initializer=init, activation='relu'
        )

    # Fully connected layer with 84 nodes
    dense2 = tf.layers.dense(
        dense1, units=84, kernel_initializer=init, activation='relu'
        )

    # Fully connected softmax output layer with 10 nodes
    dense3 = tf.layers.dense(
        dense2, units=10, kernel_initializer=init)

    # loss
    loss = tf.losses.softmax_cross_entropy(y, dense3)

    # adam optimizer
    opt = tf.train.AdamOptimizer().minimize(loss)

    # accuracy
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(dense3, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # final softmax output
    softmax = tf.nn.softmax(dense3)

    return softmax, opt, loss, accuracy
