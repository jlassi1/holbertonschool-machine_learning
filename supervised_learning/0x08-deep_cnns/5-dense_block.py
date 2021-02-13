#!/usr/bin/env python3
import tensorflow.keras as K
"""5. Dense Block"""


def dense_block(X, nb_filters, growth_rate, layers):
    """ function  that builds a dense block as described
    in Densely Connected Convolutional Networks"""
    for i in range(layers):
        conv_x = K.layers.BatchNormalization()(X)
        conv_x = K.layers.Activation('relu')(conv_x)
        conv_x = K.layers.Conv2D(
            4 * growth_rate,
            (1,
             1),
            kernel_initializer='he_normal',
            padding='same')(conv_x)
        conv_x = K.layers.BatchNormalization()(conv_x)
        conv_x = K.layers.Activation('relu')(conv_x)
        conv_x = K.layers.Conv2D(
            growth_rate,
            (3,
             3),
            kernel_initializer='he_normal',
            padding='same')(conv_x)
        X = K.layers.concatenate([X, conv_x])
        nb_filters += growth_rate

    return X, nb_filters

# def conv_layer(conv_x, filters):
#     """the bottleneck layers used for DenseNet-B"""
#     conv_x = K.layers.BatchNormalization()(conv_x)
#     conv_x = K.layers.Activation('relu')(conv_x)
#     conv_x = K.layers.Conv2D(
#         4 * filters,
#         (1,
#          1),
#         kernel_initializer='he_normal',
#         padding='same')(conv_x)
#     conv_x = K.layers.BatchNormalization()(conv_x)
#     conv_x = K.layers.Activation('relu')(conv_x)
#     conv_x = K.layers.Conv2D(
#         filters,
#         (3,
#          3),
#         kernel_initializer='he_normal',
#         padding='same')(conv_x)
#     return conv_x
