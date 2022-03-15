"""module for create_placeholders function"""
import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """creates two tf placeholders for num. examples and num. classes"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
