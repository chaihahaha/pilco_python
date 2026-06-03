# solve_chol.py
# solve_chol - solve linear equations from the Cholesky factorization.
# Solve A*X = B for X, where A is square, symmetric, positive definite. The
# input to the function is R the Cholesky decomposition of A and the matrix B.
# Example: X = solve_chol(chol(A),B);
#
# NOTE: The program code is written in the C language for efficiency and is
# contained in the file solve_chol.c, and should be compiled using matlabs mex
# facility. However, this file also contains a (less efficient) matlab
# implementation, supplied only as a help to people unfamiliar with mex. If
# the C code has been properly compiled and is avaiable, it automatically
# takes precendence over the matlab code in this file.
#
# Copyright (c) 2004, 2005, 2006 by Carl Edward Rasmussen. 2006-02-08.

# Code
import numpy as np
from scipy import linalg


def solve_chol(A, B):
    if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
        raise ValueError('Wrong sizes of matrix arguments.')

    # x = A \ (A' \ B)
    # Forward substitution: solve A' * y = B
    y = linalg.solve_triangular(A.T, B, lower=True)
    # Back substitution: solve A * x = y
    x = linalg.solve_triangular(A, y, lower=False)
    return x
