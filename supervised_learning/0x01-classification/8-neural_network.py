#!/usr/bin/env python3
"""creating a neural_network class"""


import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""
    def __init__(self, nx, nodes):
        """here start all with nx like number of input"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.ndarray((nodes, nx))
        self.W1 = np.random.normal(size=(nodes, nx))
        self.W2 = np.ndarray((1, nodes))
        self.W2[0] = np.random.normal(size=nodes)
        self.b1 = np.zeros((nodes, 1))
        self.b2 = 0
        self.A1 = 0
        self.A2 = 0
