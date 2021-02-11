#!/usr/bin/env python3
"""3. Projection Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015)"""
    # Retrieve Filters
    F11, F3, F12 = filters
    # initialize weights with he normal
    init = K.initializers.he_normal()
    # Save the input value.
    A_shortcut = A_prev
    # First component of main path
    Z = K.layers.Conv2D(
        filters=F11,
        kernel_size=(
            1,
            1),
        strides=s,
        kernel_initializer=init)(A_prev)
    Z = K.layers.BatchNormalization(axis=3)(Z)
    Z = K.layers.Activation('relu')(Z)
    # Second component of main path
    Z = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), padding='same', kernel_initializer=init)(Z)
    Z = K.layers.BatchNormalization(axis=3)(Z)
    Z = K.layers.Activation('relu')(Z)
    # Third component of main path
    Z = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), kernel_initializer=init)(Z)
    X = K.layers.BatchNormalization(axis=3)(Z)
    # shorcut path
    A_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(
            1,
            1),
        strides=s,
        padding='valid',
        kernel_initializer=init)(A_shortcut)
    A_shortcut = K.layers.BatchNormalization(axis=3)(A_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = K.layers.Add()([X, A_shortcut])
    X = K.layers.Activation('relu')(X)
    return X
