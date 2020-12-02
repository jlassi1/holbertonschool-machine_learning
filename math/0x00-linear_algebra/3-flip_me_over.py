#!/usr/bin/env python3
"""function that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """matix transpose"""
    mtx = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return mtx
