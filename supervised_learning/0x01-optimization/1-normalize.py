#!/usr/bin/env python3
"""module for normalization constants"""
import numpy as np
import tensorflow as tf


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - m)/s
