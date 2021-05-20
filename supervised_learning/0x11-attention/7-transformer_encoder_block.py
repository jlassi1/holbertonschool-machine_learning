#!/usr/bin/env python3
"""6. Transformer Encoder Block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ encoder block for a transformer """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialization"""
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def PWFFN(self):
        """ Point wise feed forward network"""
        return tf.keras.Sequential([
            self.dense_hidden,
            self.dense_output])

    def call(self, x, training, mask):
        """ call function"""
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn = self.PWFFN()
        ffn_output = ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
