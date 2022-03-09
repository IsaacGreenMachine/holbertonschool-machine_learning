import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    returns placeholders
    nx: data feature columns
    classes: num classes in classifier
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
