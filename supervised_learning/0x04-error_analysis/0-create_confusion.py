#!/usr/bin/env python3
"""0. Create Confusion """
import numpy as np


def create_confusion_matrix(labels, logits):
    """function that creates a confusion matrix"""
    m, classes = logits.shape

    confusion_m = np.zeros((classes, classes))

    for k in range(m):
        i = np.nonzero(labels[k, :])
        j = np.nonzero(logits[k, :])
        confusion_m[i, j] += 1

    return confusion_m
