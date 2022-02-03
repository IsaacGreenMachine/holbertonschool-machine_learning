#!/usr/bin/env python3
"""functions for matrix addition"""


def add_matrices2D(mat1, mat2):
    """adds two matricies"""
    if len(mat1) != len(mat2):
        return None
    else:
        new_mat = []
        for i in range(len(mat1)):
            row = add_arrays(mat1[i], mat2[i])
            if row is None:
                return None
            else:
                new_mat.append(row)
        return new_mat


def add_arrays(arr1, arr2):
    """adds two arrays"""
    if len(arr1) != len(arr2):
        return None
    else:
        new_arr = []
        for i in range(len(arr1)):
            new_arr.append(arr1[i] + arr2[i])
        return new_arr
