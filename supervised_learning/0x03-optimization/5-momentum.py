#!/usr/bin/env python3
""" 5. Momentum """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """function that updates a variable using
    the gradient descent with momentum optimization algorithm"""
    grad = beta1*v + (1-beta1)*grad
    var = var - alpha*v
    return var, grad
