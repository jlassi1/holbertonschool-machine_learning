#!/usr/bin/env python3
"""4. Moving Average """
import numpy as np


def moving_average(data, beta):
    """function  that calculates the weighted moving average of a data set"""
    n = len(data)
    zs = []
    z = 0
    for i in range(n):
        z = beta*z + (1 - beta)*data[i]
        zc = z/(1 - beta**(i+1))
        zs.append(zc)
    return zs
