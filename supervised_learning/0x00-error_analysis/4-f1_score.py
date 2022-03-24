#!/usr/bin/env python3
"""module for f1_score"""
import numpy as np


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix"""
    tru_pos = np.diag(confusion)
    fals_pos = np.sum(confusion, axis=0) - tru_pos
    fals_neg = np.sum(confusion, axis=1) - tru_pos
    precision = tru_pos / (tru_pos + fals_pos)
    recall = tru_pos / (tru_pos + fals_neg)
    return 2/((recall ** -1) + (precision ** -1))
