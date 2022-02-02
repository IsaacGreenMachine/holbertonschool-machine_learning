def matrix_shape(matrix):
    """returns a list with dimensions of given matrix"""
    # returns a list with the current dimension + all deeper dimensions.
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    # this happens when no deeper level of a matrix is available.
    except Exception:
        return []
