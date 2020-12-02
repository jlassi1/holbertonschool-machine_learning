#!/usr/bin/env python3
""" 16. The Whole Barn """
import numpy as np


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
        mat = []
        if len(mat1) == len(mat2):
            for i in range(len(mat1)):
                mat.append(mat1[i] + mat2[i])
            return mat
        return None
    result = [[mat1[i][j] + mat2[i][j] for j in range
               (len(mat1[0]))] for i in range(len(mat1))]
    return result
