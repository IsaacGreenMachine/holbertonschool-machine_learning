#!/usr/bin/env python3
"""module for sum_total function"""


def summation_i_squared(n):
    """applies summation of i^2 from 0 to n"""
    if n < 1:
        return None
    else:
        n2 = list(range(1, n+1))
        n2 = list(map(lambda n: n**2, n2))
        return sum(n2)
