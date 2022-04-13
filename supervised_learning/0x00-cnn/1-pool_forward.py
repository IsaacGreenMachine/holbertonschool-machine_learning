#!/usr/bin/env python3
"""module for pool_forward"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network

    A_prev - numpy.ndarray (m, h_prev, w_prev, c_prev). prev layer output
        m - number of examples
        h_prev - height of the previous layer
        w_prev - width of the previous layer
        c_prev - number of channels in the previous layer
    kernel_shape - tuple(kh, kw). size of the kernel for the pooling
    kh - kernel height
    kw - kernel width
    stride - tuple of (sh, sw) containing the strides for the convolution
        sh - stride for the height
        sw - stride for the width
    mode - string containing either 'max' or 'avg'

    Returns: the output of the pooling layer
    """

    im, ih, iw, ic = A_prev.shape
    kh, kw, = kernel_shape
    sh, sw = stride

    conv_size_x = ((iw - kw) // sw + 1)
    conv_size_y = ((ih - kh) // sh + 1)
    conv = np.ndarray((im, conv_size_y, conv_size_x, ic))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            if mode == "max":
                conv[:, y, x, :] = np.amax(A_prev[:,
                                                  y*sh:y*sh+kh,
                                                  x*sw:x*sw+kw,
                                                  :],
                                           axis=(1, 2))
            if mode == "avg":
                conv[:, y, x, :] = np.average(A_prev[:,
                                                     y*sh:y*sh+kh,
                                                     x*sw:x*sw+kw, :],
                                              axis=(1, 2))
    return conv
