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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A = 1 / (1 + np.exp(-1 * (np.dot(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = np.ndarray((1, X.shape[1]))
        A[0] = self.forward_prop(X)
        evaluate = np.round(A).astype(int), self.cost(Y, A)
        return evaluate
