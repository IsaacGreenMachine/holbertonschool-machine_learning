#!/usr/bin/env python3
"""module for convolve_grayscale"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images
    images - grayscale numpy.ndarray with shape (m, h, w)
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
    kernel - convolution kernel/filter numpy.ndarray with shape (kh, kw)
        kh - height of the kernel
        kw - width of the kernel
    padding:
        if ‘same’, performs a same convolution (pad 1)
        if ‘valid’, performs a valid convolution (pad 0)
        if a tuple:
            ph - padding for the height of top and bottom of image
            pw - padding for the left and right of the image
    stride - tuple of (sh, sw)
        sh - stride for the height of the image
        sw - stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images
    """
    im, ih, iw = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
        images = np.pad(images, ((0, 0),
                                 (ph, ph),
                                 (pw, pw)
                                 )
                        )
    elif padding == "same":
        ph, pw = (1, 1)
        images = np.pad(images, ((0, 0), (1, 1), (1, 1)))
    else:
        ph, pw = (0, 0)

    conv_size_x = (iw + (2 * pw) - kw) // sw + 1
    conv_size_y = (ih + (2 * ph) - kh) // sh + 1

    conv = np.ndarray((im, conv_size_y, conv_size_x))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            conv[:, y, x] = np.sum(images[:,
                                          y*sh:y*sh+kh,
                                          x*sw:x*sw+kw] * kernel,
                                   axis=(1, 2))
    return conv
