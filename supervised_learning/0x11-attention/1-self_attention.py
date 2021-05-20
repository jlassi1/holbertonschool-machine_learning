#!/usr/bin/env python3
"""1. Self Attention """
import tensorflow as tf




class SelfAttention(tf.keras.layers.Layer):
    """a class from tensorflow.keras that encode for machine translation"""
    def __init__(self, units):
        """ initialization"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """  Bahdanau implementation """
        stminus1 = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(stminus1) + self.U(hidden_states)))
        a = tf.nn.softmax(e, axis=1)
        c = a * hidden_states
        c = tf.reduce_sum(c, axis=1)
        return c, a
