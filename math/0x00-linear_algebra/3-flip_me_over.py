#!/usr/bin/env python3
"""Function that return the transpose of a matrix"""


def matrix_transpose(matrix):
    """return the transpose"""
    matrix_tran = []
    m = len(matrix)
    n = len(matrix[0])
    for i in range(0, n):
        row = []
        for j in range(0, m):
            row.append(matrix[j][i])
        matrix_tran.append(row)
    return matrix_tran
