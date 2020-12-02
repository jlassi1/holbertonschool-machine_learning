#!/usr/bin/env python3
"function that calculates the shape of a matrix"


def matrix_shape(matrix):
    "matrix shape"
    shape = [len(matrix)]
    tmp = matrix
    while isinstance(tmp[0], list):
        shape.append(len(tmp[0]))
        tmp = tmp[0]
    return shape
