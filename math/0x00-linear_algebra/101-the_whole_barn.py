#!/usr/bin/env python3
"""file for add_matrices"""


def add_matrices(mat1, mat2):
    """implements numpy add matrices"""
    m1s = matrix_shape(mat1)
    m2s = matrix_shape(mat2)
    new_mat = []
    if m1s == m2s:
        if type(mat1[0]) is int:
            return(add_arrays(mat1, mat2))
        else:
            for i in range(len(mat1)):
                new_mat.append(add_matrices(mat1[i], mat2[i]))
            return new_mat
    else:
        return None


def matrix_shape(matrix):
    """returns a list with dimensions of given matrix"""
    # returns a list with the current dimension + all deeper dimensions.
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    # this happens when no deeper level of a matrix is available.
    except Exception:
        return []


def add_arrays(arr1, arr2):
    """adds two arrays"""
    if len(arr1) != len(arr2):
        return None
    else:
        new_arr = []
        for i in range(len(arr1)):
            new_arr.append(arr1[i] + arr2[i])
        return new_arr
