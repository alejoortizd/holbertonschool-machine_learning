#!/usr/bin/env python3
"""Calculate accuracy"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate accuracy"""
    accuracy = tf.argmax(y_pred, 1)
    equal = tf.equal(tf.argmax(y, 1), accuracy)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
