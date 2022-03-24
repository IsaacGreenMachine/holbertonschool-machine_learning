#!/usr/bin/env python3
"""module for precision"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix"""
    sum = np.sum(confusion, axis=0)
    tru_pos = np.diag(confusion)
    fals_pos = sum - tru_pos
    return tru_pos / (tru_pos + fals_pos)
