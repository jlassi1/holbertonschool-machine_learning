#!/usr/bin/env python3
""" 4.train  """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ Function that trains a model using mini-batch
        gradient descent """
    history = network.fit(x=data, y=labels, epochs=epochs, verbose=verbose,
                          batch_size=batch_size, shuffle=shuffle)
    return history
