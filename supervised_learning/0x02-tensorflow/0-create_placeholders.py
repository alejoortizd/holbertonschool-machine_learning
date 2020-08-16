#!/usr/bin/env python3
"""
function def create_placeholders(nx, classes): that returns two placeholders
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """Start the function"""
    return(tf.placeholder(float, shape=[None, nx], name='x'),
           tf.placeholder(float, shape=[None, classes], name='y'))
