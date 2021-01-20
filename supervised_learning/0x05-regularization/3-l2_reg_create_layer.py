#!/usr/bin/env python3
""" 2. L2 Regularization Cost """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    functionthat creates a tensorflow layer that includes L2 regularization
    """
    kernel_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel_init,
                            kernel_regularizer=l2,
                            name="layer",
                            )
    return layer(prev)
