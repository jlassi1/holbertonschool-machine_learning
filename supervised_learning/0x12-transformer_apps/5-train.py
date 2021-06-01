#!/usr/bin/env python3
""" Transformer Trainning"""
import tensorflow as tf
import numpy as np
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


def train_step(inp, tar, transformer, optimizer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(
            inputs=inp,
            target=tar_inp,
            training=True,
            encoder_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            decoder_mask=dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """creates and trains a transformer model for
    machine translation of Portuguese to English"""

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    data = Dataset(batch_size, max_len)
    vocab_size_pt = data.tokenizer_pt.vocab_size + 2
    vocab_size_en = data.tokenizer_en.vocab_size + 2
    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        vocab_size_pt,
        vocab_size_en,
        max_len,
        max_len)

    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar, transformer, optimizer)

            if batch % 50 == 0:
                print (
                    'Epoch {} Batch {} Loss {} Accuracy {}'.format(
                        epoch + 1,
                        batch,
                        train_loss.result(),
                        train_accuracy.result()))

        print ('Epoch {} Loss {} Accuracy {}'.format(epoch + 1,
                                                     train_loss.result(),
                                                     train_accuracy.result()))
