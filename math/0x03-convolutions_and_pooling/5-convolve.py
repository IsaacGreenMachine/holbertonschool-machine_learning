#!/usr/bin/env python3
"""module for convolve"""
import numpy as np
from math import ceil, floor


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on images with channels:

    images - numpy.ndarray with shape (m, h, w, c) containing multiple images
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
        c - number of channels in the image
    kernels - numpy.ndarray with shape (kh, kw, c, nc)
        kh - height of a kernel
        kw - width of a kernel
        c - number of channels in the image
        nc - number of kernels
    padding - tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    the image is padded with 0’s

    stride - tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """
    im, ih, iw, ic = images.shape
    kh, kw, kc, knc = kernels.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding.shape
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    elif padding == "same":
        ph, pw = (1, 1)
        images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    else:
        ph, pw = (0, 0)

    conv_size_x = floor(((iw - kw + (2 * pw))/sw) + 1)
    conv_size_y = floor(((ih - kh + (2 * ph))/sh) + 1)
    conv = np.ndarray((im, conv_size_y, conv_size_x, knc))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            for k in range(kernels.shape[3]):
                conv[:, y, x, k] = np.sum(images[:,
                                                 y*sh:y*sh+kh,
                                                 x*sw:x*sw+kw,
                                                 0:kc] * kernels[:, :, :, k],
                                          axis=(1, 2, 3))
    return conv
