#!/usr/bin/env python3
""" 2. Identity Block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """function that builds an identity block as described
    in Deep Residual Learning for Image Recognition (2015)"""
    # Retrieve Filters
    F11, F3, F12 = filters
    # initialize weights with he normal
    init = K.initializers.he_normal()
    # Save the input value.
    A_shortcut = A_prev
    # First component of main path
    Z = K.layers.Conv2D(F11, (1, 1), strides=(
        1, 1), kernel_initializer=init)(A_prev)
    Z = K.layers.BatchNormalization(axis=3)(Z)
    Z = K.layers.Activation('relu')(Z)
    # Second component of main path
    Z = K.layers.Conv2D(
        F3, kernel_size=(
            3, 3), strides=(
            1, 1), padding='same', kernel_initializer=init)(Z)
    Z = K.layers.BatchNormalization(axis=3)(Z)
    Z = K.layers.Activation('relu')(Z)
    # Third component of main path
    Z = K.layers.Conv2D(
        F12, kernel_size=(
            1, 1), strides=(
            1, 1), padding='valid', kernel_initializer=init)(Z)
    X = K.layers.BatchNormalization(axis=3)(Z)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    X = K.layers.Add()([A_shortcut, X])
    X = K.layers.Activation('relu')(X)

    return X
