#!/usr/bin/env python3
""" 10. Transformer Networ"""
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Transformer(tf.keras.Model):
    """ a transformer network """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        """ initialization"""
        super(Transformer, self).__init__()

        self.tokenizer = EncoderBlock(
            N, dm, h, hidden,
            input_vocab, max_seq_input, drop_rate)

        self.decoder = DecoderBlock(
            N, dm, h, hidden,
            target_vocab, max_seq_target, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """ call function """
        enc_output = self.tokenizer(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.final_layer(dec_output)

        return final_output
