#!/usr/bin/env python3
""" One-Hot Decode """
import numpy as np


def one_hot_decode(one_hot):
    """function that converts a one-hot matrix into a vector of labels"""
    try:
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
