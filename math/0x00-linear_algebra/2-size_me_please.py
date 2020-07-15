#!/usr/bin/env python3
"""Here we are going to create a function that calculate the shape"""


def shape_matrix(matrix, a):
    """Function to calculate shape right now"""
    if type(matrix) != list:
        return a
    else:
        a.append(len(matrix))
        if type(matrix) == list and len(matrix) > 0:
            shape_matrix(matrix[0], a)


def matrix_shape(matrix):
    """Functio that call another function to calculate the shape"""
    a = []
    shape_matrix(matrix, a)
    return a
