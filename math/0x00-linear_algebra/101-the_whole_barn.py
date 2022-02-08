#!/usr/bin/env python3
"""file for add_matrices"""


def add_matrices(mat1, mat2):
    """implements numpy add matrices"""
    pass
    # if m1.shape == m2.shape:
    #    return (m1 + m2).tolist()
    # else:
    #    return None
    '''
    mat1s = matrix_shape(mat1)
    mat2s = matrix_shape(mat2)
    if mat1s == mat2s:
        for i

    else:
        return None
'''


def matrix_shape(matrix):
    """returns a list with dimensions of given matrix"""
    # returns a list with the current dimension + all deeper dimensions.
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    # this happens when no deeper level of a matrix is available.
    except Exception:
        return []
