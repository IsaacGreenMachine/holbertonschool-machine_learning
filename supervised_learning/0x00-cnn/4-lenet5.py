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

    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = tf.layers.conv2d(x, 6, 5, padding='same')

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = tf.layers.conv2d(pool1, 16, 5, padding='valid')

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    pool2_flat = tf.layers.flatten(pool2)

    # initializer for dense layers
    # he_norm = tf.initializers.he_normal()
    he_norm = tf.initializers.variance_scaling()

    # Fully connected layer with 120 nodes
    dense1 = tf.layers.dense(pool2_flat,
                             120,
                             activation='relu',
                             kernel_initializer=he_norm)

    # Fully connected layer with 84 nodes
    dense2 = tf.layers.dense(dense1,
                             84,
                             activation='relu',
                             kernel_initializer=he_norm)

    # Fully connected softmax output layer with 10 nodes
    dense3 = tf.layers.dense(dense2,
                             10,
                             activation='softmax',
                             kernel_initializer=he_norm)

    # loss
    loss = tf.losses.softmax_cross_entropy(y, dense3)

    # adam optimizer
    opt = tf.train.AdamOptimizer()

    # accuracy
    y_out = tf.argmax(y, axis=1)
    pred_out = tf.argmax(dense3, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.math.equal(y_out, pred_out), tf.float32))

    return dense3, opt.minimize(loss), loss, acc
