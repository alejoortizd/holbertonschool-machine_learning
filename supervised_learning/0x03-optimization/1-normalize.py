#!/usr/bin/env python3
"""normalize function"""


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix
    """
    return (X - m) / s
