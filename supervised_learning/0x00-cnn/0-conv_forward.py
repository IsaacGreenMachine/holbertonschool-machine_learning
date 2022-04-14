#!/usr/bin/env python3
"""module for conv_forward"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward prop over a convolutional layer of a neural network

    A_prev - numpy.ndarray (m, h_prev, w_prev, c_prev). prev layer output
        m - number of examples
        h_prev - height of the previous layer
        w_prev - width of the previous layer
        c_prev - number of channels in the previous layer

    W - numpy.ndarray (kh, kw, c_prev, c_new). kernels
        kh - filter height
        kw - filter width
        c_prev - number of channels in the previous layer
        c_new - number of channels in the output

    b - numpy.ndarray(1, 1, 1, c_new). biases

    activation - activation function applied to the convolution

    padding - string. "same" or "valid", indicating the type of padding used

    stride - tuple of (sh, sw) containing the strides for the convolution
        sh - stride for the height
        sw - stride for the width

    Returns: the output of the convolutional layer
    """
    im, ih, iw, ic = A_prev.shape
    kh, kw, kc, knc = W.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == "same":
        ph = (((ih - 1) * sh) + kh - ih) // 2
        pw = (((iw - 1) * sw) + kw - iw) // 2
    elif padding == "valid":
        ph, pw = 0, 0

    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    conv_size_x = (iw - kw + (2 * pw)) // sw + 1
    conv_size_y = (ih - kh + (2 * ph)) // sh + 1

    conv = np.ndarray((im, conv_size_y, conv_size_x, knc))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            for k in range(W.shape[3]):
                conv[:, y, x, k] = np.sum(padded[:,
                                                 y*sh:y*sh+kh,
                                                 x*sw:x*sw+kw,
                                                 0:kc] * W[:, :, :, k],
                                          axis=(1, 2, 3))
    return activation(conv + b)
