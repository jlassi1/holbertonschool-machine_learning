#!/usr/bin/env python3
def matrix_transpose(matrix):
    "function that returns the transpose of a 2D matrix"
    mtx = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))] 
    return mtx