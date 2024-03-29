#!/usr/bin/env python3
""" 5. Definiteness """
import numpy as np


def definiteness(matrix):
    """function that calculates the definiteness of a matrix"""
    definitess = [
        'Positive definite',
        'Positive semi-definite',
        'Negative semi-definite',
        'Negative definite',
        'Indefinite']
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if not np.array_equal(matrix.T, matrix):
        return None
    try:
        w, v = np.linalg.eig(matrix)
        if all(w > 0):
            return definitess[0]
        elif all(w >= 0):
            return definitess[1]
        if all(w < 0):
            return definitess[3]
        elif all(w <= 0):
            return definitess[2]
        else:
            return definitess[4]
    except Exception:
        return None
