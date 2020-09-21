#!/usr/bin/env python3
"""Creates the training operation for a neural network
in tensorflow using the Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    opt = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return opt
