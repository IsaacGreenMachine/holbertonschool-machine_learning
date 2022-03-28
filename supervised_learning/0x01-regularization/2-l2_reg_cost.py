#!/usr/bin/env python3
"""module for l2_reg_cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates cost of nn with L2 regularization
    the main file sets up each layer with l2 reg, tf.get_reg_loss will
    get the l2 losses of the network


    cost - tensor containing the cost of network without L2 regularization

    Returns - tensor containing the cost of the network with L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
