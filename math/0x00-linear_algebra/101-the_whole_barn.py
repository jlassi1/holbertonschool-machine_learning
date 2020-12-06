#!/usr/bin/env python3
""" 16. The Whole Barn """


def matrix_shape(matrix):
    "matrix shape"
    shape = [len(matrix)]
    tmp = matrix
    while isinstance(tmp[0], list):
        shape.append(len(tmp[0]))
        tmp = tmp[0]
    return shape


def add_matrices(mat1, mat2):
    """ function that add two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if isinstance(mat1[0], int):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    else:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
