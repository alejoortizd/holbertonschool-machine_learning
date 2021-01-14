#!/usr/bin/env python3
"""creating a neuron class"""


import numpy as np


class Neuron:
    """
    class Neuron that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """here start all with nx like number of input"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.ndarray((1, nx))
        self.W[0] = np.random.normal(size=nx)
        self.b = 0
        self.A = 0
