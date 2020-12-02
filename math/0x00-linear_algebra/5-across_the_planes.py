#!/usr/bin/env python3
""" Across The Planes """


def add_matrices2D(mat1, mat2):
    """ function that adds two matrices element-wise """
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        mat = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
        return mat
    else:
        return None
