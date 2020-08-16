#!/usr/bin/env python3
"""
function def create_layer(prev, n, activation):
"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """Start the function"""
    initializer = (tf.contrib.layers.
                   variance_scaling_scaling_initializer(mode="FAN_AVG"))
    return tf.layers.Dense(n, activation, name='layer',
                           kernel_initializer=initializer)(prev)
