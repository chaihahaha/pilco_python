# checkgrad.py
# *Summary:* checkgrad checks the derivatives in a function, by comparing them
# to finite differences approximations. The partial derivatives and the
# approximation are printed and the norm of the difference divided by the
# norm of the sum is returned as an indication of accuracy.
#
#    (d, dy, dh) = checkgrad(f, X, e, *args)
#
#
# *Input arguments:*
#
#   f          callable that returns (fX, dfX) = f(X, *args)
#              where fX is the function value and dfX is the gradient of fX
#              with respect to the parameters X
#   X          parameters (vector or dict)
#   e          small perturbation used for finite differences (1e-4 is good)
#   *args      other arguments that are passed on to the function f
#
#
# *Output arguments:*
#
#   d          relative error of analytical vs. finite difference gradient
#   dy         analytical gradient
#   dh         finite difference gradient
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-21
#
# High-Level Steps
# # Analytical gradient
# # Numerical gradient via finite differences
# # Relative error

import numpy as np

from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def checkgrad(f, X, e, *args):
    ## Code

    # 1. Analytical gradient
    Z = unwrap(X)                                     # number of input variables
    NZ = len(Z)

    y, dy = f(X, *args)                               # get the partial derivatives dy

    y = np.asarray(y, dtype=float)
    if y.ndim == 0:
        D, E_dim = 1, 1                               # scalar
    elif y.ndim == 1:
        D, E_dim = len(y), 1                          # column vector
    else:
        D, E_dim = y.shape
    y = y.ravel(order='F')
    Ny = len(y)                                       # number of output variables

    if isinstance(dy, (dict, list, tuple)):
        dy = unwrap(dy)
    dy = np.asarray(dy, dtype=float).ravel(order='F')
    dy = dy.reshape(Ny, NZ, order='F')

    # 2. Finite difference approximation
    dh = np.zeros((Ny, NZ))
    for j in range(NZ):
        dx = np.zeros(NZ)
        dx[j] = dx[j] + e                             # perturb a single dimension
        y2, _ = f(rewrap(X, Z + dx)[0], *args)
        y1, _ = f(rewrap(X, Z - dx)[0], *args)
        dh[:, j] = (np.asarray(y2, dtype=float).ravel(order='F') -
                    np.asarray(y1, dtype=float).ravel(order='F')) / (2 * e)

    # 3. Compute error
    # norm of diff divided by norm of sum
    sum_sq = np.sum((dh + dy)**2, axis=1)
    # handle zero divisors
    sum_sq[sum_sq == 0] = np.inf
    d = np.sqrt(np.sum((dh - dy)**2, axis=1) / sum_sq)
    small = np.max(np.abs(np.column_stack([dy, dh])), axis=1) < 1e-5  # small derivatives are poorly tested ...
    d[(d > 1e-3) & small] = np.nan                                       # ... by finite differences
    d = d.reshape(D, E_dim, order='F')

    print('   Analytic  Numerical')
    for i in range(Ny):
        if NZ == 1:
            print(f'  {dy[i, :]} \t {dh[i, :]}')
        else:
            print(np.column_stack([dy[i, :], dh[i, :]]))                 # print the two vectors
        idx = np.unravel_index(i, (D, E_dim), order='F')
        print(f'd = {d[idx]:e}\n')

    if Ny > 1 and (D > 1 or E_dim > 1):
        print('For all outputs, d = ')
        print(d)
    print()

    return d, dy, dh
