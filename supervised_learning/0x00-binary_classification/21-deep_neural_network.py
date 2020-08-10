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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for x in range(self.__L):
            if type(layers[x]) is not int or layers[x] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wx = 'W'+str(x + 1)
            bx = 'b'+str(x + 1)
            if x == 0:
                self.__weights[wx] = np.random.randn(layers[x], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.__weights[wx] = np.random.randn(layers[x], layers[x-1])\
                                   * np.sqrt(2/layers[x-1])
            self.__weights[bx] = np.zeros((layers[x], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward propagation'''
        self.__cache['A0'] = X
        for layer in range(1, self.__L + 1):
            xi = self.__cache['A'+str(layer-1)]
            z = np.dot(self.__weights['W'+str(layer)], xi) +\
                self.__weights['b'+str(layer)]

            sigmoid = 1 / (1 + np.exp(-z))
            self.__cache['A'+str(layer)] = sigmoid

        return sigmoid, self.__cache

    def cost(self, Y, A):
        """
        Calculate cost of neural network
        """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """
        Evaluate the neural network
        """
        A = self.forward_prop(X)[0]
        return A.round().astype(int), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent on the neural network"""
        m = len(Y[0])
        dz = cache['A'+str(self.__L)] - Y
        for layer in range(self.__L, 0, -1):
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(cache['A'+str(layer-1)], dz.T)
            dz = np.matmul(self.__weights['W'+str(layer)].T, dz) *\
                (cache['A'+str(layer-1)] * (1 - cache['A'+str(layer-1)]))
            self.__weights['W'+str(layer)] = self.__weights['W'+str(layer)] -\
                (alpha * dW).T
            self.__weights['b'+str(layer)] = self.__weights['b'+str(layer)] -\
                (alpha * db)
