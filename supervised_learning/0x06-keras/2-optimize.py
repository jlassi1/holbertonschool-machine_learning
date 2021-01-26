#!/usr/bin/env python3
""" 2. Optimize  """
from tensorflow import keras as K


def optimize_model(network, alpha, beta1, beta2):
    """function that sets up Adam optimization for a keras model
    with categorical crossentropy loss and accuracy metrics"""
    optimation = K.optimizers.Adam(
        lr=alpha,
        beta_1=beta1,
        beta_2=beta2)
    network.compile(optimation,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
