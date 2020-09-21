#!/usr/bin/env python3
"""shuffle data function"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    """
    randomize = np.random.permutation(X.shape[0])
    input1 = X[randomize]
    input2 = Y[randomize]
    return input1, input2
