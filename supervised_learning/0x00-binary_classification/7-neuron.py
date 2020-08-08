#!/usr/bin/env python3
"""creating a neuron class"""


import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        self.__W[0] = (self.__W[0] - alpha *
                       np.dot(X, (A - Y).T).T[0] / X.shape[1])
        self.__b -= alpha * (A[0] - Y[0]).mean()

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Training a neuron"""
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
        perdida = []
        plots = []
        while counter < iterations:
            A = self.forward_prop(X)
            if verbose and not (counter % step):
                print("Cost after {} iterations: {}"
                      .format(counter, self.cost(Y, A)))
                if graph:
                    perdida.append(self.cost(Y, A))
                    plots.append(counter)
            self.gradient_descent(X, Y, A, alpha)
            counter += 1
        self.forward_prop(X)
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(counter, self.cost(Y, self.__A)))
        if graph:
            plt.plot(plots, perdida, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
        return self.evaluate(X, Y)
