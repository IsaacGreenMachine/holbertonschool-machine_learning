"""module for calculate_accuracy function"""
import numpy as np
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """creates tf graph struct for calculating accuracy"""
    ymax = tf.argmax(y, axis=1)
    predmax = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.math.equal(ymax, predmax),
                          tf.float32))
