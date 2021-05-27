#!/usr/bin/env python3
""" Dataset for machine translation """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """initialization"""
        (data_train, data_valid), metadata = tfds.load(
          'ted_hrlr_translate/pt_to_en',
          split=['train', 'validation'],
          as_supervised=True, with_info=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
          data_train)
        data_train = data_train.map(self.tf_encode)
        train_buffer_size = metadata.splits['train'].num_examples

        def filter_max_length(x, y, max_length=max_len):
            """ filtering out sentences with length > max_length"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)
        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        data_train = data_train.shuffle(train_buffer_size).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(filter_max_length)
        self.data_valid = data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers for our dataset"""
        tokenize = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_en = tokenize(
            [en.numpy() for _, en in data], target_vocab_size=2**15)
        tokenizer_pt = tokenize(
            [pt.numpy() for pt, _ in data], target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = [vocab_size_pt] + \
            self.tokenizer_pt.encode(pt.numpy()) + \
            [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + \
            self.tokenizer_en.encode(en.numpy()) + \
            [vocab_size_en + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """acts as a tensorflow wrapper for the encode instance method """
        tf_pt, tf_en = tf.py_function(
            self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])

        tf_pt.set_shape([None])
        tf_en.set_shape([None])
        return tf_pt, tf_en
