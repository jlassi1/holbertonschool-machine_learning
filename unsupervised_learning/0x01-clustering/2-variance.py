#!/usr/bin/env python3
""" 2. Variance """
import numpy as np


def dist(data, centers):
    """function that calcules the distant between centre and data"""
    distance = np.sqrt(np.sum(
        (np.array(centers) - data[:, None, :])**2, axis=2))
    return distance


def variance(X, C):
    """function that calculates the total
    intra-cluster variance for a data set"""
    try:
        closest = np.argmin(dist(X, C), axis=1)
        var = np.linalg.norm(X - C[closest]) ** 2
        return var
    except Exception:
        return None
