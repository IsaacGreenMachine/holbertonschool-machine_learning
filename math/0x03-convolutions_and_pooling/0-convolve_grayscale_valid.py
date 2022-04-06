#!/usr/bin/env python3
"""module for convolve_grayscale_valid"""
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
    conv_size = images.shape[1]-kernel.shape[0]+1
    conv = np.zeros((images.shape[0], conv_size, conv_size))

    for x in range(conv_size):
        for y in range(conv_size):
            conv[:, x, y] = np.sum(images[:,
                                          x:x+kernel.shape[0],
                                          y:y+kernel.shape[1]
                                          ] * kernel,
                                   axis=(1, 2))
    return conv
    '''
    samples, samp_h, samp_w = images.shape
    filter_h, filter_w = kernel.shape

    dim_h = samp_h - filter_h + 1
    dim_w = samp_w - filter_w + 1

    conv = np.zeros((samples, dim_h, dim_w))

    for w in range(dim_h):
        for h in range(dim_w):
            conv[:, w, h] = (
                kernel * images[:, w: w+filter_w, h:h+filter_h]
                ).sum(axis=(1, 2))
    return conv
