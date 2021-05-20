#!/usr/bin/env python3
"""4. Scaled Dot Product Attention """
import tensorflow as tf


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
