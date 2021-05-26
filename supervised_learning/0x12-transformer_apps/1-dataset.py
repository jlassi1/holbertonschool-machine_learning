#!/usr/bin/env python3
""" Dataset for machine translation """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """initialization"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """function that creates sub-word tokenizers for our dataset"""
        subword_tok = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        token_en = subword_tok(
            [en.numpy() for _, en in data], target_vocab_size=2**15)
        token_pt = subword_tok(
            [pt.numpy() for pt, _ in data], target_vocab_size=2**15)
        return token_pt, token_en

    def encode(self, pt, en):
        """ encodes a translation into tokens """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = [vocab_size_pt] + \
            self.tokenizer_pt.encode(pt.numpy().decode('utf-8')) + \
            [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + \
            self.tokenizer_en.encode(en.numpy().decode('utf-8')) + \
            [vocab_size_en + 1]
        return pt_tokens, en_tokens
