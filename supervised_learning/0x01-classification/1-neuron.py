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
        self.__W = np.ndarray((1, nx))
        self.__W[0] = np.random.normal(size=nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights of private W"""
        return self.__W

    @property
    def b(self):
        """The activated output of the neuron. of private b"""
        return self.__b

    @property
    def A(self):
        """The bias for the neuron. of private b"""
        return self.__A
