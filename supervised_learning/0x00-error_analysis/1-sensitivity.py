#!/usr/bin/env python3
"""module for sensitivity"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    sum = np.sum(confusion, axis=1)
    tru_pos = np.diag(confusion)
    fals_neg = sum - tru_pos
    return tru_pos / (tru_pos + fals_neg)
