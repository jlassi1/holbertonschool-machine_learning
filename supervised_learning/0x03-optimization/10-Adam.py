#!/usr/bin/env python3
""" 10. Adam Upgraded """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """function that creates the training operation for
    a neural network in tensorflow using
    the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
