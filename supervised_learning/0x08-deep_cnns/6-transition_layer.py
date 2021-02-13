#!/usr/bin/env python3
"""6. Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ function that  builds a transition layer as described
    in Densely Connected Convolutional Networks"""
    out_channels = int(nb_filters * compression)
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(out_channels, (1, 1),
                        kernel_initializer='he_normal', padding='same')(x)
    x = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x, out_channels
