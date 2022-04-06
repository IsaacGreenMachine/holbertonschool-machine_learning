#!/usr/bin/env python3
"""module for pool"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images:
    images - numpy.ndarray with shape (m, h, w, c) containing multiple images
        m - number of images
        h - height in pixels of the images
        w - width in pixels of the images
        c - number of channels in the image
    kernel_shape - tuple of (kh, kw)
        kh - height of the kernel
        kw - width of the kernel
    stride - tuple of (sh, sw)
        sh - stride for the height of the image
        sw - stride for the width of the image
    mode - type of pooling
        'max' indicates max pooling
        'avg' indicates average pooling


    Returns: a numpy.ndarray containing the pooled images
    """
    im, ih, iw, ic = images.shape
    kh, kw, = kernel_shape
    sh, sw = stride

    conv_size_x = ((iw - kw) // sw + 1)
    conv_size_y = ((ih - kh) // sh + 1)
    conv = np.ndarray((im, conv_size_y, conv_size_x, ic))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            if mode == "max":
                conv[:, y, x, :] = np.amax(images[:,
                                                  y*sh:y*sh+kh,
                                                  x*sw:x*sw+kw,
                                                  :],
                                           axis=(1, 2))
            if mode == "avg":
                conv[:, y, x, :] = np.average(images[:,
                                                     y*sh:y*sh+kh,
                                                     x*sw:x*sw+kw, :],
                                              axis=(1, 2))
    return conv
