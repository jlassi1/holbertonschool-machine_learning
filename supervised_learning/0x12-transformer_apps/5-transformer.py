#!/usr/bin/env python3
""" Transformer Networ"""
import tensorflow as tf
import numpy as np


def sdp_attention(Q, K, V, mask=None):
    """function that calculates the scaled dot product attention"""

    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights


def get_angles(pos, i, d_model):
    """function that calculetes the angle for the positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """function that calculates the positional encoding for a transformer"""

    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads


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


class Encoder(tf.keras.layers.Layer):
    """create the encoder for a transformer"""
    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len, drop_rate=0.1):
        """initialization"""
        super(Encoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """ call function """
        seq_len = x.shape[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x


class DecoderBlock(tf.keras.layers.Layer):
    """ decoder block for a transformer """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialization"""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def PWFFN(self):
        """ Point wise feed forward network"""
        return tf.keras.Sequential([
            self.dense_hidden,
            self.dense_output])

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ call function"""

        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn = self.PWFFN()
        ffn_output = ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class Decoder(tf.keras.layers.Layer):
    """decoder for a transformer"""
    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len, drop_rate=0.1):
        """initialization """
        super(Decoder, self).__init__()

        self.dm = dm
        self.N = N

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ call function """

        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x)

        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.Model):
    """ a transformer network """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """ initialization"""
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            N, dm, h, hidden,
            input_vocab, max_seq_input, drop_rate)

        self.decoder = Decoder(
            N, dm, h, hidden,
            target_vocab, max_seq_target, drop_rate)

        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """ call function """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
