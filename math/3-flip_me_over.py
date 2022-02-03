#!/usr/bin/env python3
"""file for matrix_transpose function"""


def matrix_transpose(matrix):
    """returns a transpose of a given matrix"""
    fin_mat = []
    # height of matrix (inside lists)
    for i in range(len(matrix[0])):
        row = []
        # width of matrix (outside lists)
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        fin_mat.append(row)
    return fin_mat
