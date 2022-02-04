#!/usr/bin/env python3
"""returns middle columns of a python 2D Array"""
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for i in range(len(matrix)):
    the_middle.append(matrix[i][2:4])
print("The middle columns of the matrix are: {}".format(the_middle))
