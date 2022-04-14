#!/usr/bin/env python3
"""module for pool_backward"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network

    dA - numpy.ndarray(m, h_new, w_new, c_new).
    (partial derivatives with respect to pooling output layer)
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels

    A_prev - numpy.ndarray of shape (m, h_prev, w_prev, c).
    (output of the previous layer)
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c is the number of channels

    kernel_shape - tuple of (kh, kw). kernel size for pooling
        kh is the kernel height
        kw is the kernel width

    stride - tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width

    mode - string "max" or "avg", whether to perform max or avg pooling.

    Returns: partial derivatives with respect to the previous layer (dA_prev)
    """

    m, hP, wP, cP = dA.shape
    _, hN, wN, cN = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # set up output shape full of zeros
    der = np.zeros_like(A_prev)

    # for each example
    for frame in range(m):
        # for each row (move vertically)
        for h in range(hP):
            # current spot vertically
            ah = sh * h
            # for each column (move horizontally)
            for w in range(wP):
                # current spot horizontally
                aw = sw * w
                # for each channel
                for c in range(cP):

                    if mode == 'avg':
                        # average of filter-sized chunk at (h, w)
                        avg_dA = dA[frame, h, w, c] / kh / kw
                        # adding average to each value in current area
                        der[frame, ah: ah+kh, aw: aw+kw, c] += (
                            np.ones((kh, kw)) * avg_dA
                        )

                    if mode == 'max':
                        # area to choose from
                        box = A_prev[frame, ah: ah+kh, aw: aw+kw, c]
                        # create OH matrix where only 1 is max value from box
                        mask = (box == np.max(box))
                        # add value from dA to max value from box
                        der[frame, ah: ah+kh, aw: aw+kw, c] += (
                            mask * dA[frame, h, w, c]
                        )
    return der
