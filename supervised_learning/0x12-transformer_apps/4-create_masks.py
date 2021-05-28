#!/usr/bin/env python3

""" 4. Create Masks """
import tensorflow.compat.v2 as tf


def create_look_ahead_mask(shape=(0, 0, 0, 0)):
    """The look-ahead mask is used to mask the future tokens in a sequence.
    In other words, the mask indicates which entries should not be used."""
    mask = 1 - tf.linalg.band_part(tf.ones(shape), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    """ add extra dimensions to add the padding
    to the attention logits"""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    batch_size, seq_len_out = target.shape
    batch_size, seq_len_in = inputs.shape

    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    look_ahead_mask = create_look_ahead_mask(
        shape=(batch_size, 1, seq_len_out, seq_len_out))

    decoder_target_padding_mask = create_padding_mask(target)
    """takes the maximum between
    look_ahead_mask and decoder target padding mask"""
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
