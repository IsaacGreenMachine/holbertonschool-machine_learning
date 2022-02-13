#!/usr/bin/env python3
"""module for poly_integral function"""


def poly_integral(poly, C=0):
    """integrates a polynomial"""
    if type(poly) is list and type(C) is int and len(poly) > 0:
        new_list = []
        new_list.append(C)
        for i in range(1, len(poly)+1):
            num = poly[i - 1]/i
            if num % 1 == 0:
                new_list.append(int(num))
            else:
                new_list.append(num)
        while new_list[-1] == 0 and len(new_list) > 1:
            new_list.pop(-1)
        return new_list
    else:
        return None
