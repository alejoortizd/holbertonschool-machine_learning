#!/usr/bin/env python3
"""Updates a variable using the gradient
descent with momentum optimization algorithm"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient
    descent with momentum optimization algorithm"""
    V = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * V)
    return W, V