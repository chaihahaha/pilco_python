# lossQuad.py
# *Summary:* Compute expectation and variance of a quadratic cost
# $(x-z)'*W*(x-z)$
# and their derivatives, where $x \sim N(m,S)$
#
#
#  function [L, dLdm, dLds, S, dSdm, dSds, C, dCdm, dCds] = lossQuad(cost, m, S)
#
#
#
# *Input arguments:*
#
#    cost
#      .z:     target state                                              [D x 1]
#      .W:     weight matrix                                             [D x D]
#    m         mean of input distribution                                [D x 1]
#    s         covariance matrix of input distribution                   [D x D]
#
#
# *Output arguments:*
#
#   L               expected loss                                  [1   x    1 ]
#   dLdm            derivative of L wrt input mean                 [1   x    D ]
#   dLds            derivative of L wrt input covariance           [1   x   D^2]
#   S               variance of loss                               [1   x    1 ]
#   dSdm            derivative of S wrt input mean                 [1   x    D ]
#   dSds            derivative of S wrt input covariance           [1   x   D^2]
#   C               inv(S) times input-output covariance           [D   x    1 ]
#   dCdm            derivative of C wrt input mean                 [D   x    D ]
#   dCds            derivative of C wrt input covariance           [D   x   D^2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-30
#
# High-Level Steps
# # Expected cost
# # Variance of cost
# # inv(s)* cov(x,L)

import numpy as np


def lossQuad(cost, m, S, compute_derivatives=True):
    ## Code
    D = len(m)  # get state dimension

    # ensure m is a column vector
    m = np.atleast_1d(np.asarray(m)).reshape(-1, 1).astype(float)

    # set some defaults if necessary
    if 'W' in cost:
        W = cost['W']
    else:
        W = np.eye(D)
    if 'z' in cost:
        z = cost['z']
    else:
        z = np.zeros((D, 1))
    # ensure z is a column vector
    z = np.atleast_1d(np.asarray(z)).reshape(-1, 1).astype(float)

    # 1. expected cost
    L = np.dot(S.ravel(), W.ravel()) + (z - m).T @ W @ (z - m)
    L = L.item() if isinstance(L, np.ndarray) else L

    # 1a. derivatives of expected cost
    dLdm = 2 * (m - z).T @ W       # wrt input mean
    dLds = W.T                     # wrt input covariance matrix

    # 2. variance of cost
    # NOTE: MATLAB original assigns this to variable S, which shadows the input
    # covariance matrix S. This causes dSdm and dSds to use the scalar variance
    # instead of the covariance matrix, producing incorrect derivatives.
    # This Python translation preserves the input covariance S and stores the
    # output variance in S_val, giving correct derivatives.
    Sw = W + W.T
    S_val = np.trace(W @ S @ Sw @ S) + (z - m).T @ Sw @ S @ Sw @ (z - m)
    S_val = S_val.item() if isinstance(S_val, np.ndarray) else S_val
    if S_val < 1e-12:
        S_val = 0.0                # for numerical reasons

    # 2a. derivatives of variance of cost
    # wrt input mean
    dSdm = -(2 * Sw @ S @ (W + W) @ (z - m)).T
    # wrt input covariance matrix
    dSds = W.T @ S.T @ Sw.T + Sw.T @ S.T @ W.T + Sw @ (z - m) @ (Sw @ (z - m)).T

    # 3. inv(s) times IO covariance with derivatives
    C = 2 * W @ (m - z)
    dCdm = 2 * W
    dCds = np.zeros((D, D * D))

    return L, dLdm, dLds, S_val, dSdm, dSds, C, dCdm, dCds
