#!/usr/bin/env python3
"""deep neural network class"""


import numpy as np
import pickle


class DeepNeuralNetwork:
    """deep neural network class"""

    def __init__(self, nx, layers):
        """
        Start the function
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {"W1": np.random.randn(layers[0], nx) *
                          np.sqrt(2 / nx),
                          "b1": np.zeros((layers[0], 1))}
        for layer, i in enumerate(layers[1:], 2):
            actual = "W" + str(layer)
            self.__weights[actual] = (np.random.randn(i, layers[layer - 2]) *
                                      np.sqrt(2 / layers[layer - 2]))
            actual = "b" + str(layer)
            self.__weights[actual] = np.zeros((layers[layer - 1], 1))

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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1:
                raise ValueError("step must be a positive integer")
        counter = 0
        loss = []
        x = []
        while counter < iterations:
            A, cache = self.forward_prop(X)
            if verbose and not (counter % step):
                print("Cost after {} iterations: {}"
                      .format(counter, self.cost(Y, A)))
            if graph:
                loss.append(self.cost(Y, A))
                x.append(counter)
            self.gradient_descent(Y, cache, alpha)
            counter += 1
        self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}".format(
                counter, self.cost(Y, self.__cache["A" + str(self.__L)])))
        if graph:
            plt.plot(x, loss, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """Save class to .pkl"""
        if len(filename) < 4 or filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        """Load a DeepNeuralNetwork from file"""
        try:
            with open(filename, "rb") as infile:
                return pickle.load(infile)
        except FileNotFoundError:
            return None

    @staticmethod
    def one_hot_encode(Y, classes):
        """Convert a numeric label vector to a one-hot matrix"""
        onehot = np.zeros((classes, Y.shape[0]))
        for ex, label in enumerate(Y):
            onehot[label][ex] = 1
        return onehot
