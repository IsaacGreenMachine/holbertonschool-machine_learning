#!/usr/bin/env python3
"""module for sum_total function"""


def summation_i_squared(n):
    """applies summation of i^2 from 0 to n"""
    if n < 1:
        return None
    else:
        sum = 0
        for i in range(1, n+1):
            sum += (i)**2
        return sum
