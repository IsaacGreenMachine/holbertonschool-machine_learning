#!/usr/bin/env python3
"""module for convolve_channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on images with channels:

    images - numpy.ndarray with shape (m, h, w, c) containing multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel - kernel numpy.ndarray with shape (kh, kw, c)
        kh is the height of the kernel
        kw is the width of the kernel
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
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding.shape
        images = np.pad(images, ((0, 0), (pw, pw), (ph, ph)))
    elif padding == "same":
        ph = (((ih - 1) * sh) + kh - ih) // 2 + 1
        pw = (((iw - 1) * sw) + kw - iw) // 2 + 1
        images = np.pad(images, ((0, 0), (pw, pw), (ph, ph)))
    else:
        ph, pw = (0, 0)

    conv_size_x = (iw - kw + (2 * pw)) // sw + 1
    conv_size_y = (ih - kh + (2 * ph)) // sh + 1

    conv = np.ndarray((im, conv_size_y, conv_size_x))

    for x in range(conv_size_x):
        for y in range(conv_size_y):
            conv[:, y, x] = np.sum(images[:,
                                          y*sh:y*sh+kh,
                                          x*sw:x*sw+kw,
                                          0:kc] * kernel,
                                   axis=(1, 2, 3))
    return conv
