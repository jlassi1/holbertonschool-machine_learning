#!/usr/bin/env python3
""" 6. Train """
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """ Learning Rate Decay """
    callback = None
    if early_stopping:
        callback = K.callbacks.EarlyStopping(patience=patience)
    history = network.fit(x=data, y=labels, callbacks=[callback],
                          epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
