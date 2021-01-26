#!/usr/bin/env python3
""" 5. Train """
from tensorflow import keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                verbose=True,
                shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    return network.fit(data, labels, nb_epoch=epochs,
                       batch_size=batch_size, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle)