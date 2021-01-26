#!/usr/bin/env python3
""" 7. Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """ learning rate decay"""
    def scheduler(epoch):
        """ scheduler function """
        return alpha / (1 + decay_rate * epoch)
    callback = []
    if early_stopping:
        callback.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))
    if save_best:
        callback.append(K.callbacks.ModelCheckpoint(
            filepath, save_best_only=True))
    history = network.fit(x=data, y=labels, callbacks=callback,
                          epochs=epochs, batch_size=batch_size,
                          validation_data=validation_data,
                          verbose=verbose, shuffle=shuffle)
    return history
