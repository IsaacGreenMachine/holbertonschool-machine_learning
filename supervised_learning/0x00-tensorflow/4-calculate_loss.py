"""module for calculate_loss function"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """creates tensor for calculating loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
