"""module for create_train_op function"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates tensor to train with gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
