#!/usr/bin/env python3
"""module for poly_derivative function"""


def poly_derivative(poly):
    """applies derivative to a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        new_poly = []
        for i in range(1, len(poly)):
            new_poly.append(poly[i] * i)
        return new_poly
