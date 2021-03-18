#!/usr/bin/env python3
""" 3. Adjugate  """


def determinant(matrix):
    """function that calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]] or len(matrix[0]) == 0:
        return 1
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1:
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
        det += ((-1)**i) * matrix[0][i] * determinant(x)
    return det


def cofactor(matrix):
    """function that calculates the cofactor  matrix of a matrix"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if not all(isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix) or matrix == []:
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix[0]) == 1 or matrix == [[]] or len(matrix[0]) == 0:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]
    mi = []
    n = len(matrix)
    y = 0
    for j in range(n):
        r = []
        for i in range(n):
            x = [[matrix[i][j] for j in range(n)] for i in range(n)]
            x.pop(j)
            for m in range(n - 1):
                x[m].pop(i)
            y = ((-1)**(i + j)) * determinant(x)
            r.append(y)
        mi.append(list(r))
    return mi


def adjugate(matrix):
    """function that calculates the adjugate of a matrix"""
    m = cofactor(matrix)
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
