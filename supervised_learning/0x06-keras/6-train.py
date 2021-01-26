#!/usr/bin/env python3
""" 6. Train """
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """ early stooping """
    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(monitor='loss', patience=patience)
    history = network.fit(x=data, y=labels, callbacks=[callback],
                          epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
