#!/usr/bin/env python3
"""module for conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of a neural network

    dZ - numpy.ndarray(m, h_new, w_new, c_new)
        partial derivatives with respect to the
        unactivated output of the convolutional layer
        m - number of examples
        h_new - height of the output
        w_new - width of the output
        c_new - number of channels in the output

    A_prev - numpy.ndarray(m, h_prev, w_prev, c_prev). output of prev layer
        m - number of examples
        h_prev - height of the previous layer
        w_prev - width of the previous layer
        c_prev - number of channels in the previous layer

    W - numpy.ndarray(kh, kw, c_prev, c_new). kernels for the convolution
        kh - filter height
        kw - filter width
        c_prev - number of channels in the previous layer
        c_new - number of channels in the output

    b - numpy.ndarray(1, 1, 1, c_new). biases applied to the convolution
        c_new - number of channels in the output

    padding - string. "same" or "valid". type of padding used

    stride - tuple(sh, sw). strides for the convolution
        sh - stride for the height
        sw - stride for the width

    Returns: partial derivatives with respect to prev layer (dA_prev),
            the kernels (dW),
            the biases (db)
    """

    _, hP, wP, cP = A_prev.shape
    m, hN, wN, cN = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    dW = np.zeros_like(W)
    dA = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((hP - 1) * sh) + kh - hP) // 2 + 1
        pad_w = (((wP - 1) * sw) + kw - wP) // 2 + 1

    # padding A_prev based on padding type
    A_prev = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
        )

    # padding dA based on padding type
    dA = np.pad(dA, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant', constant_values=0)

    # for each item in training set
    for frame in range(m):
        # for each pixel vertically in dZ
        for h in range(hN):
            # for each pixel horizontally in dZ
            for w in range(wN):
                # for each channel in dZ
                for flt in range(cN):
                    # applies convolution of A_prev and dZ
                    dW[:, :, :, flt] += np.multiply(
                        A_prev[frame, h*sh:h*sh+kh, w*sw:w*sw+kw, :],
                        dZ[frame, h, w, flt]
                    )
                    # applies convolution of W and dZ
                    dA[frame, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += (
                        np.multiply(W[:, :, :, flt], dZ[frame, h, w, flt])
                    )
    # final padding adjustments
    if padding == 'same':
        dA = dA[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA, dW, db
