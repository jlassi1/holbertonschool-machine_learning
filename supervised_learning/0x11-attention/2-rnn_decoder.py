#!/usr/bin/env python3
"""2. RNN Decoder  """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """a class from tensorflow.keras that encode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """ initialization """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer="glorot_uniform",
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """call funtion """
        context_vector, attention_w = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)

        return x, state
