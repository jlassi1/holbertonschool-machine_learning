#!/usr/bin/env python3
""" Layers  """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ function that create layers """
    kernel_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=kernel_init, name="layer",
                            )
    return layer
