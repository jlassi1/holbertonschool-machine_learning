#!/usr/bin/env python3
""" 1. Input """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    l2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2)(x)

        if i < len(layers)-1:
            x = K.layers.Dropout((1-keep_prob))(x)

    return K.Model(inputs=inputs, outputs=x)
