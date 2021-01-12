#!/usr/bin/env python3
""" 1. Normalize"""
import numpy as np


def normalize(X, m, s):
    """function that normalizes (standardizes) a matrix"""
    return (X - m) / s
