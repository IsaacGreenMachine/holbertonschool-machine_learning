#!/usr/bin/env python3
"""module for convolve_grayscale_padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:
    images - grayscale numpy.ndarray with shape (m, h, w)
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
    kernel - convolution kernel/filter numpy.ndarray with shape (kh, kw)
        kh - height of the kernel
        kw - width of the kernel
    padding is a tuple of (ph, pw)
        ph - padding for the height of top and bottom of image
        pw - padding for the left and right of the image
    the image is padded with 0’s
    uses two loops

    Returns: a numpy.ndarray containing the convolved images
    """

    conv_size_x = images.shape[2] - kernel.shape[1] + (2 * padding[1]) + 1
    conv_size_y = images.shape[1] - kernel.shape[0] + (2 * padding[0]) + 1
    conv = np.ndarray((images.shape[0], conv_size_y, conv_size_x))
    images = np.pad(images, ((0, 0),
                             (padding[0], padding[0]),
                             (padding[1], padding[1]))
                    )
    for x in range(conv_size_x):
        for y in range(conv_size_y):
            conv[:, y, x] = np.sum(images[:,
                                          y:y+kernel.shape[0],
                                          x:x+kernel.shape[1]] * kernel,
                                   axis=(1, 2))
    return conv
