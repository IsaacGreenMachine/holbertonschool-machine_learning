#!/usr/bin/env python3
"""Module containing the function train.
"""

import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a neural network classifier.

    Args:
        X_train (numpy.ndarray): N-dimensional array containing the training
            input data.
        Y_train (numpy.ndarray): N-dimensional array containing the training
            labels.
        X_valid (numpy.ndarray): N-dimensional array containing the validation
            input data.
        Y_valid (numpy.ndarray): N-dimensional array containing the validation
            labels.
        layer_sizes (List): List containing the number of nodes in each layer
            of the network.
        activations (List): List containing the activation functions for each
            layer of the network.
        alpha (float): The learning rate
        iterations (int): The number of iterations to train over.
        save_path (str, optional): Designates where to save the model.
            Defaults to "/tmp/model.ckpt".

    Returns:
        The path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    store = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    for i in range(iterations + 1):
        train_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sess.run(accuracy,
                                  feed_dict={x: X_train, y: Y_train})
        valid_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        if i % 100 == 0 or i == iterations:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
        if i < iterations:
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

    return store.save(sess, save_path)
