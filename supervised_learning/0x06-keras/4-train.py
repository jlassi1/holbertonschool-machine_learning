#!/usr/bin/env python3
""" 4. Train """
import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                verbose=True,
                shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    History = network.fit(data, labels, nb_epoch=epochs,
                          batch_size=batch_size,
                          verbose=verbose, shuffle=shuffle)
    return History
