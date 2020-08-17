#!/usr/bin/env python3
"""Create training op"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """Create training op"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
