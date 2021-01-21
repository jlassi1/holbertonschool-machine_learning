#!/usr/bin/env python3
""" 2. L2 Regularization Cost """


import tensorflow as tf


def l2_reg_cost(cost):
    """
    function that calculates the cost
    of a neural network with L2 regularization
    """
    loss = cost + tf.losses.get_regularization_loss()
    return loss
