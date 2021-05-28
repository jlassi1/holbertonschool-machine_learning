#!/usr/bin/env python3
""" Train transformer """
import tensorflow as tf
import numpy as np
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer model for
    machine translation of Portuguese to English"""
    data = Dataset(batch_size, max_len)
    pt, en = data.data_train
    encoder_mask, combined_mask, decoder_mask = create_masks(pt, en)
    transformer = Transformer(
        N, dm, h, hidden, input_vocab,
        en.vocab_size, max_seq_input, max_seq_target)
