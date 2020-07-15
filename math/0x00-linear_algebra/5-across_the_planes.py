#!/usr/bin/env python3
"""function that return the add of 2 matrices"""


def add_matrices2D(mat1, mat2):
    """return add of 2 matrices"""
    addmat = []
    if (len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0])):
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            addmat.append(row)
        return addmat
    return None
