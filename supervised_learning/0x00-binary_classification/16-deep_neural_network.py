#!/usr/bin/env python3
"""Deep neural network class"""


import numpy as np


class DeepNeuralNetwork:
    """Deep neural network"""

    def __init__(self, nx, layers):
        """init the class"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for x in range(self.L):
            if type(layers[x]) is not int or layers[x] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wx = 'W'+str(x + 1)
            bx = 'b'+str(x + 1)
            if x == 0:
                self.weights[wx] = np.random.randn(layers[x], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.weights[wx] = np.random.randn(layers[x], layers[x-1])\
                                   * np.sqrt(2/layers[x-1])
            self.weights[bx] = np.zeros((layers[x], 1))
