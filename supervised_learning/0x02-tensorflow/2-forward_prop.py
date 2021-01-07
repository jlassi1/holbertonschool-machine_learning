#!/usr/bin/env python3
""" Forward Propagation  """
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ function that creates the forward propagation
    graph for the neural network"""
    layers = x
    for j in range(len(layer_sizes)):
        layers = create_layer(layers, layer_sizes[j],
                              activation=activations[j])
    return layers
