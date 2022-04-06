#!/usr/bin/env python3
"""module for convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    adds padding to image so that output is of same size as input

    Returns: numpy.ndarray containing the convolved images
    """
    conv_size_h = images.shape[1]-kernel.shape[0]+1
    conv_size_w = images.shape[2]-kernel.shape[1]+1
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    conv = np.zeros((images.shape[0], conv_size_h, conv_size_w))
    for x in range(conv_size_h):
        for y in range(conv_size_w):
            conv[:, x, y] = np.sum(images[:,
                                          x:x+kernel.shape[0],
                                          y:y+kernel.shape[1]] * kernel,
                                   axis=(1, 2))
    return conv
