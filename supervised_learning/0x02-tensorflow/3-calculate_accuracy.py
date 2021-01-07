#!/usr/bin/env python3
"""  Accuracy  """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction"""
    prediction = tf.math.round(y_pred)
    equality = tf.math.equal(prediction, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
