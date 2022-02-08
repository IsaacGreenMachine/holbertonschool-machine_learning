#!/usr/bin/env python3
"""file for np_slice function"""


def np_slice(matrix, axes={}):
    """slices numpy matrix of any dimension"""
    sliceShape = []
    for ax in range(len(matrix.shape)):
        indicies = (axes.get(ax))
        if type(indicies) is int:
            sliceShape.append(slice(None, indicies, None))
        elif indicies is None or len(indicies) == 0:
            sliceShape.append(slice(None, None, None))
        elif len(indicies) == 1:
            sliceShape.append(slice(indicies[0], None, None))
        elif len(indicies) == 2:
            sliceShape.append(slice(indicies[0], indicies[1], None))
        elif len(indicies) == 3:
            sliceShape.append(slice(indicies[0], indicies[1], indicies[2]))
    return matrix[tuple(sliceShape)]
