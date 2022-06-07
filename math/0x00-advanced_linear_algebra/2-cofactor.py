#!/usr/bin/env python3
"""module for cofactor function"""


def cofactor(matrix):
    """returns the cofactor of a matrix"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    width = len(matrix)
    if width == 0:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(item) != width:
            raise ValueError("matrix must be a non-empty square matrix")

    ################
    new_mat = minor(deepcopy(matrix))
    modfirst = 1
    modcurr = 1
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            # print(matrix[row][col], end=' ')
            new_mat[row][col] *= (modfirst * modcurr)
            modcurr *= -1
        modfirst *= -1
        modcurr = 1
        # print()
    return new_mat
    ################


def deepcopy(matrix):
    """returns a deep copy of a matrix (no pointers/links at all)"""
    new_matrix = []
    for row in range(len(matrix)):
        new_matrix.append([])
        for col in range(len(matrix[row])):
            new_matrix[row].append(matrix[row][col])
    return new_matrix


def determinant(matrix):
    """returns the determinant of a given matrix"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    width = len(matrix)
    if width == 0:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(item) != width and width != 1:
            raise ValueError("matrix must be a non-empty square matrix")
    if width == 1:
        if len(matrix[0]) == 0:
            raise 1
        elif len(matrix[0]) == 1:
            return matrix[0][0]
        else:
            raise ValueError("matrix must be a non-empty square matrix")
    else:
        if width == 2:
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


def minor(matrix):
    """returns the minor of a matrix"""
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    width = len(matrix)
    if width == 0:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(item) != width:
            raise ValueError("matrix must be a non-empty square matrix")
    if width == 1:
        return [[1]]
    elif width == 2:
        return (
          [[int(matrix[1][1]), int(matrix[1][0])],
           [int(matrix[0][1]), int(matrix[0][0])]]
          )
    else:
        new_mat = deepcopy(matrix)
        for i in range(len(matrix[0])):
            for j in range(len(matrix[0])):
                sub_mat = []
                for row in range(len(matrix[0])):
                    sub_mat.append([])
                    for col in range(len(matrix[0])):
                        if row != i and col != j:
                            sub_mat[row].append(int(matrix[row][col]))
                sub_mat = [x for x in sub_mat if x != []]
                new_mat[i][j] = determinant(sub_mat)
        return new_mat
