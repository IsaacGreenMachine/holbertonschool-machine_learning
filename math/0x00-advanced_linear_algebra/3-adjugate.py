#!/usr/bin/env python3
"""module for adjugate function"""


def adjugate(matrix):
    """returns the adjugate of a matrix"""
    if (
        matrix and matrix[0] and type(matrix) is list and
        all(type(sub) is list for sub in matrix)
    ):
        width = len(matrix)
        if matrix == [[]]:
            raise ValueError("matrix must be a non-empty square matrix")
        for height in matrix:
            if len(height) != width:
                raise ValueError("matrix must be a non-empty square matrix")
    ################
        return(matrix_transpose(cofactor(matrix)))
    ################
    else:
        raise TypeError("matrix must be a list of lists")


def cofactor(matrix):
    """returns the cofactor of a matrix"""
    if (
        matrix and matrix[0] and type(matrix) is list and
        all(type(sub) is list for sub in matrix)
    ):
        width = len(matrix)
        if matrix == [[]]:
            raise ValueError("matrix must be a non-empty square matrix")
        for height in matrix:
            if len(height) != width:
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

    else:
        raise TypeError("matrix must be a list of lists")


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
    if matrix == [[]]:
        return 1
    if (
        matrix and matrix[0] and type(matrix) is list
        and all(type(sub) is list for sub in matrix)
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


def minor(matrix):
    """returns the minor of a matrix"""
    if (
            matrix and matrix[0] and type(matrix) is list and
            all(type(sub) is list for sub in matrix)):
        width = len(matrix)
        if matrix == [[]]:
            raise ValueError("matrix must be a non-empty square matrix")
        for height in matrix:
            if len(height) != width:
                raise ValueError("matrix must be a non-empty square matrix")
    ################
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
    ################
    else:
        raise TypeError("matrix must be a list of lists")


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
