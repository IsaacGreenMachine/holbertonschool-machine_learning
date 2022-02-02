def mat_mul(mat1, mat2):
    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None
    else:
        new_mat = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat2[0])):
                val = 0
                for k in range(len(mat2)):
                    val += mat1[i][k] * mat2[k][j]
                row.append(val)
            new_mat.append(row.copy())
        return new_mat


def matrix_shape(matrix):
    """returns a list with dimensions of given matrix"""
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    except Exception:
        return []
