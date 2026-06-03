# lossLin.py
# *Summary:* Function to compute the expected loss and its derivatives, given an 
# input distribution, under a linear loss function: L = a^T(x - b). Note, this
# loss function can return negative loss.
#
#   def lossLin(cost, m, s, compute_derivatives=True)
#
# *Input arguments:*
#
#   cost
#     .a      gradient of linear loss function,                       [D x 1]
#     .b      targets, the value of x for which there is zero loss    [D x 1]
#   m         mean of input distribution                              [D x 1]
#   s         covariance matrix of input distribution                 [D x D]
#
# *Output arguments:*
#
#  L               expected loss                                  [1   x    1 ]
#  dLdm            derivative of L wrt input mean                 [1   x    D ]
#  dLds            derivative of L wrt input covariance           [1   x   D^2]
#  S               variance of loss                               [1   x    1 ]
#  dSdm            derivative of S wrt input mean                 [1   x    D ]
#  dSds            derivative of S wrt input covariance           [1   x   D^2]
#  C               inv(S) times input-output covariance           [D   x    1 ]
#  dCdm            derivative of C wrt input mean                 [D   x    D ]
#  dCds            derivative of C wrt input covariance           [D   x   D^2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-05
#
## High-Level Steps
# # Expected cost
# # Variance of cost
# # inv(s)* cov(x,L)

import numpy as np


def lossLin(cost, m, s, compute_derivatives=True):
    ## Code

    a = np.atleast_2d(cost['a']).reshape(-1, 1)
    b = np.atleast_2d(cost['b']).reshape(-1, 1)
    D = len(m)

    if len(a) != D or len(b) != D:
        raise ValueError('a or b not the same length as m')

    m = np.atleast_2d(m).reshape(-1, 1)

    # 1) Mean
    L = float(a.T @ (m - b))
    dLdm = a.T
    dLds = np.zeros((D, D))

    S_var = None
    dSdm = None
    dSds = None
    C = None
    dCdm = None
    dCds = None

    if not compute_derivatives:
        return L, dLdm, dLds, S_var, dSdm, dSds, C, dCdm, dCds

    # 2) Variance
    S_var = float(a.T @ s @ a)
    dSdm = np.zeros((1, D))
    dSds = a @ a.T

    # 3) inv(s) * input-output covariance cov(x,L)
    C = a.copy()
    dCdm = np.zeros((D, D))
    dCds = np.zeros((D, D * D))

    return L, dLdm, dLds, S_var, dSdm, dSds, C, dCdm, dCds
