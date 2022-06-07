#!/usr/bin/env python3
"""module for definiteness function"""
import numpy as np


def definiteness(matrix):
    """returns the definiteness of a matrix"""
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    dets = []
    for i in range(1, matrix.shape[0] + 1):
        dets.append(np.linalg.det(matrix[:i, :i]))
    if all(x > 0 for x in dets):
        return "Positive definite"
    if all(x < 0 for x in dets[::2]) and all(x > 0 for x in dets[1::2]):
        return "Negative definite"
    elif all(x >= 0 for x in dets):
        return "Positive semi-definite"

    elif all(x <= 0 for x in dets):
        return "Negative semi-definite"
    else:
        return "Indefinite"
