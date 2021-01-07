#!/usr/bin/env python3
"""  Accuracy  """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction"""
    prediction = tf.math.argmin(y_pred, axis=1)
    equality = tf.math.equal(prediction, tf.math.argmin(y, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
