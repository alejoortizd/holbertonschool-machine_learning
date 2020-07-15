#!/usr/bin/env python3
"""Function that multiplicate 2 matrices"""


def mat_mul(mat1, mat2):
    """Function that return a multiplicate of matrices"""
    res = [[]]
    # copy data
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    # know row the math1 and columns the mat2
    mat1n = len(a[0])
    mat2m = len(b)
    # validate m == n
    if mat1n == mat2m:
        # create zip object to manipulate better the data
        res = [[sum(a*b for a, b in zip(i, j)) for j in zip(*b)] for i in a]
        return res
    return None
