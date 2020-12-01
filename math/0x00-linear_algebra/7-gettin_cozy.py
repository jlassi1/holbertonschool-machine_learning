#!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    "function that concatenates two matrices along a specific axis"
    mat = None
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        mat = [x for x in mat1] + [y for y in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        mat = [x + y for x, y in zip(mat1, mat2)]
    return mat

