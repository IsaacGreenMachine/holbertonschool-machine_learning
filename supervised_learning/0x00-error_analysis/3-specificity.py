#!/usr/bin/env python3
"""module for specificity"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    tru_pos = np.diag(confusion)
    fals_pos = np.sum(confusion, axis=0) - tru_pos
    fals_neg = np.sum(confusion, axis=1) - tru_pos
    tru_neg = np.sum(confusion) - (tru_pos + fals_neg + fals_pos)
    return tru_neg / (tru_neg + fals_pos)
