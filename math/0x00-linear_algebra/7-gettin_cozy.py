#!/usr/bin/env python3
"""file for cat_matrices2D function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concats two matrices"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            new_mat = []
            for row in mat1:
                new_mat.append(row.copy())
            for i in range(len(mat2)):
                new_mat.append(mat2[i].copy())
            return new_mat
    else:
        if len(mat1) != len(mat2):
            return None
        else:
            new_mat = []
            for i in range(len(mat1)):
                new_mat.append(mat1[i].copy() + mat2[i].copy())
            return new_mat
