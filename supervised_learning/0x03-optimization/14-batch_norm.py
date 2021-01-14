#!/usr/bin/env python3
""" 14. Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """functionthat creates a batch normalization layer
    for a neural network in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init, name="layer"
                            )
    m, s = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    normal = tf.nn.batch_normalization(layer(prev), m, s, beta, gamma, 1e-8)

    return activation(normal)
