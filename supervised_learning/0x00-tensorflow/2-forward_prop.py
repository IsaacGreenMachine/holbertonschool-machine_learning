"""module for forward_prop function"""
import numpy as np
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates neural network for forward prop"""
    create_layer = __import__('1-create_layer').create_layer
    prev = x
    for i in range(len(layer_sizes)):
        lay = create_layer(prev, layer_sizes[i], activations[i])
        prev = lay(prev)
    return prev
