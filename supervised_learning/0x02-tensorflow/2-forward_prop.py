#!/usr/bin/env python3
"""basic forward propagation network"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation function"""
    initial_layer = create_layer(x, layer_sizes[0], activations[0])
    final_layer = initial_layer
    for layer in range(1, len(layer_sizes)):
        final_layer = create_layer(final_layer, layer_sizes[layer],
                                   activations[layer])
    return final_layer
