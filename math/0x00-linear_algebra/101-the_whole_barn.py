#!/usr/bin/env python3
""" 16. The Whole Barn """
import numpy as np


def add_matrices(mat1, mat2):
    """ function that add two matrices"""
    if np.shape(mat1) != np.shape(mat2):
        return None
    return np.add(mat1, mat2)
