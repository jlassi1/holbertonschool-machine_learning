#!/usr/bin/env python3
"""4. F1 score """

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix:"""
    """ specificity = TN/(TN+FP)"""
    """ f1_score = harmonic mean of the precision and recall"""

    prec = precision(confusion)
    recall = sensitivity(confusion)
    f1 = 2 * prec * recall / (prec + recall)
    return f1
