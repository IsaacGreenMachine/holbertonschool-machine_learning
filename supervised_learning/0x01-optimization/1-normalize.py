#!/usr/bin/env python3
"""module for normalization constants"""


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - m)/s
