#!/usr/bin/env python3
""" identity_block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ Function that builds an identity block as
        described in Deep Residual Learning for Image
        Recognition (2015) """
    F11, F3, F12 = filters
    he_init = K.initializers.he_normal()
    conv_1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=he_init)(A_prev)
    norm_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    Act1 = K.layers.Activation('relu')(norm_1)
    conv_2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=he_init)(Act1)
    norm_2 = K.layers.BatchNormalization(axis=3)(conv_2)
    Act2 = K.layers.Activation('relu')(norm_2)
    conv_3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=he_init)(Act2)
    norm_3 = K.layers.BatchNormalization(axis=3)(conv_3)
    add1 = K.layers.Add()([norm_3, A_prev])
    res = K.layers.Activation('relu')(add1)
    return res
