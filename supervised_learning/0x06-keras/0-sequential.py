#!/usr/bin/env python3
""" 0. Sequential  """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    model = K.Sequential()
    l2 = K.regularizers.l2(l=lambtha)
    model.add(K.layers.Dense(
        layers[0], input_dim=nx,
        activation=activations[0], kernel_regularizer=l2))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1-keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=l2))
    return model
