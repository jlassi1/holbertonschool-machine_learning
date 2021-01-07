#!/usr/bin/env python3
""" One-Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric
    label vector into a one-hot matrix"""
    try:
        n_Y = np.max(Y) + 1
        return np.eye(n_Y)[Y].T
    except Exception:
        return None
