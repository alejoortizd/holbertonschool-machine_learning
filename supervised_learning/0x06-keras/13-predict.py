#!/usr/bin/env python3
"""Makes a prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    if verbose:
        pred = network.predict(data, verbose=1)
    else:
        pred = network.predict(data, verbose=0)
    return pred
