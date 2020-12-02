#!/usr/bin/env python3
"""  13. Cat's Got Your Tongue """

def np_cat(mat1, mat2, axis=0):
    "function that concatenates two matrices along a specific axis"
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis)
