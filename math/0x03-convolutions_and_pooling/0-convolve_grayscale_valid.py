#!/usr/bin/env python3
"""module for convolve_grayscale_valid(images, kernel)"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images

    images - grayscale numpy.ndarray with shape (m, h, w)
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
    kernel - convolution kernel/filter numpy.ndarray with shape (kh, kw)
        kh - height of the kernel
        kw - width of the kernel
    uses two for loops.

    Returns: numpy.ndarray containing the convolved images
    """

    '''
    conv_size_h = images.shape[1]-kernel.shape[0]+1
    conv_size_w = images.shape[2]-kernel.shape[1]+1
    conv = np.zeros((images.shape[0], conv_size_h, conv_size_w))

    for x in range(conv_size_w):
        for y in range(conv_size_h):
            conv[:, x, y] = np.sum(images[:,
                                          x:x+kernel.shape[1],
                                          y:y+kernel.shape[0]
                                          ] * kernel,
                                   axis=(1, 2))
    return conv

    '''
    im, ih, iw = images.shape
    kh, kw = kernel.shape

    dim_h = ih - kh + 1
    dim_w = iw - kw + 1

    conv = np.zeros((im, dim_h, dim_w))

    for w in range(dim_h):
        for h in range(dim_w):
            conv[:, w, h] = (
                kernel * images[:, w: w+kw, h:h+kh]
                ).sum(axis=(1, 2))
    return conv
