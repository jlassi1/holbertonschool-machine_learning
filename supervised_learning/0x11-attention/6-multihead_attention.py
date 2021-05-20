#!/usr/bin/env python3
"""5. Multi Head Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """class to perform multi head attention """
    def __init__(self, dm, h):
        """initialization"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth).
        Transpose the result such that
        the shape is (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """call function  """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)  # (batch_size, seq_len, dm)
        k = self.Wk(K)  # (batch_size, seq_len, dm)
        v = self.Wv(V)  # (batch_size, seq_len, dm)

        # (batch_size, h, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, h, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, h, seq_len_q, depth)
        # attention_Weights.shape == (batch_size, h, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdp_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, dm)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # (batch_size, seq_len_q, dm)
        output = self.linear(concat_attention)

        return output, attention_weights
