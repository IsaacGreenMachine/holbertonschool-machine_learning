#!/usr/bin/env python3
"""module for determinant function"""


def determinant(matrix):
    """returns the determinant of a given matrix"""
    if matrix == [[]]:
        return 1
    if (
      matrix and matrix[0] and type(matrix) is list
      and type(matrix[0]) is list
      ):
        width = len(matrix)
        for height in matrix:
            if len(height) != width:
                raise ValueError("matrix must be a square matrix")

    ################
        if width == 1:
            return matrix[0][0]
        elif width == 2:
            return (
              (matrix[0][0] * matrix[1][1]) -
              (matrix[0][1] * matrix[1][0])
              )
        else:
            det = 0
            for i in range(len(matrix[0])):
                sub_mat = [[]]
                for row in range(len(matrix[0])):
                    for col in range(len(matrix[0])):
                        if row != 0 and col != i:
                            sub_mat[row].append(matrix[row][col])
                    sub_mat.append([])
                sub_mat.pop(0)
                sub_mat.pop()
                if i == 0 or i % 2 == 0:
                    det += (matrix[0][i] * determinant(sub_mat))
                else:
                    det -= (matrix[0][i] * determinant(sub_mat))
            return det
    ################

    else:
        raise TypeError("matrix must be a list of lists")
