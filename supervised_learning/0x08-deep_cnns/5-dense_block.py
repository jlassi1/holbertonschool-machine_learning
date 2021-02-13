#!/usr/bin/env python3
"""5. Dense Block"""
import tensorflow.keras as K


def conv_layer(conv_x, filters):
    """the bottleneck layers used for DenseNet-B"""
    conv_x = K.layers.BatchNormalization()(conv_x)
    conv_x = K.layers.Activation('relu')(conv_x)
    conv_x = K.layers.Conv2D(
        4 * filters,
        (1,
         1),
        kernel_initializer='he_normal',
        padding='same')(conv_x)
    conv_x = K.layers.BatchNormalization()(conv_x)
    conv_x = K.layers.Activation('relu')(conv_x)
    conv_x = K.layers.Conv2D(
        filters,
        (3,
         3),
        kernel_initializer='he_normal',
        padding='same')(conv_x)
    return conv_x


def dense_block(X, nb_filters, growth_rate, layers):
    """ function  that builds a dense block as described
    in Densely Connected Convolutional Networks"""
    for i in range(layers):
        each_layer = conv_layer(X, growth_rate)
        X = K.layers.concatenate([X, each_layer])
        nb_filters += growth_rate

    return X, nb_filters
