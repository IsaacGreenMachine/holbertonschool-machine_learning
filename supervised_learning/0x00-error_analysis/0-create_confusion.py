#!/usr/bin/env python3
"""module for create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a labels x logits confusion matrix"""
    confusion = np.zeros([labels.shape[1], logits.shape[1]])
    lab_idx, log_idx = np.where(labels == 1)[1], np.where(logits == 1)[1]
    idxs = list(zip(lab_idx, log_idx))
    unique, counts = np.unique(idxs, return_counts=True, axis=0)
    confusion[unique[:, 0], unique[:, 1]] = counts
    return confusion
