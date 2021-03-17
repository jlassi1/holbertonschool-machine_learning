#!/usr/bin/env python3
""" 0. Determinant  """
import copy


def determinant(matrix):
    """function that calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if all(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    n = len(matrix)
    for i in range(n):
        x = [[matrix[i][j] for j in range(n)] for i in range(n)]
        x.pop(0)
        for m in range(n - 1):
            x[m].pop(i)
        det += (-1)**i * matrix[0][i] * determinant(x)
    return det
