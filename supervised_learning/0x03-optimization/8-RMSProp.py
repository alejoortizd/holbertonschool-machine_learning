#!/usr/bin/env python3
"""Creates the training operation for a neural network
in tensorflow using the RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm"""
    r = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon).minimize(loss)
    return r