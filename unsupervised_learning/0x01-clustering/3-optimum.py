#!/usr/bin/env python3
"""3. Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """function that tests for the optimum number of clusters by variance"""
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin - 1:
        return None, None
    try:
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if k == kmin:
                var_min = variance(X, C)
            results.append((C, clss))
            var = variance(X, C)
            d_vars.append(var_min - var)
    except Exception:
        return None, None
    return results, d_vars
