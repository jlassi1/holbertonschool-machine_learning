#!/usr/bin/env python3
""" Placeholders  """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ function that create placeholders X, Y """
    x = tf.placeholder(tf.float32, [None, nx], name="x")
    y = tf.placeholder(tf.float32, [None, classes], name="y")
    return x, y
