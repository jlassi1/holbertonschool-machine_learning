#!/usr/bin/env python3
def matrix_shape(matrix):
    "function that calculates the shape of a matrix"
    shape = [len(matrix)]
    tmp = matrix
    while isinstance(tmp[0], list):
        shape.append(len(tmp[0]))
        tmp = tmp[0]
    return shape