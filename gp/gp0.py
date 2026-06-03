
"""
gp0.py
*Summary:* Compute joint predictions for multiple GPs with uncertain inputs.
If gpmodel.nigp exists, individial noise contributions are added.
Predictive variances contain uncertainty about the function, but no noise.

  function [M, S, V] = gp0(gpmodel, m, s)

*Input arguments:*

  gpmodel    GP model struct
    hyp      log-hyper-parameters                                  [D+2 x  E ]
    inputs   training inputs                                       [ n  x  D ]
    targets  training targets                                      [ n  x  E ]
    nigp     (optional) individual noise variance terms            [ n  x  E ]
  m          mean of the test distribution                         [ D  x  1 ]
  s          covariance matrix of the test distribution            [ D  x  D ]

*Output arguments:*

  M          mean of pred. distribution                            [ E  x  1 ]
  S          covariance of the pred. distribution                  [ E  x  E ]
  V          inv(s) times covariance between input and output      [ D  x  E ]


Copyright (C) 2008-2013 by
Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.

Last modified: 2013-05-24

High-Level Steps
# If necessary, compute kernel matrix and cache it
# Compute predicted mean and inv(s) times input-output covariance
# Compute predictive covariance matrix, non-central moments
# Centralize moments
"""

import numpy as np
from scipy.linalg import cho_solve

from pilco_python.util.maha import maha


_cache = {'K': None, 'iK': None, 'beta': None, 'oldX': None, 'oldn': None}


def _get_field_gp0(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]


def _has_field_gp0(obj, key):
    if hasattr(obj, 'get'):
        return key in obj
    return hasattr(obj, key)


def gp0(gpmodel, m, s, compute_derivatives=False):
    global _cache

    inputs = _get_field_gp0(gpmodel, 'inputs')
    targets = _get_field_gp0(gpmodel, 'targets')
    X = _get_field_gp0(gpmodel, 'hyp')
    n, D = inputs.shape
    n2, E = targets.shape
    if n != n2:
        raise ValueError('inputs and targets must have same number of rows')

    # 1) if necessary: re-compute cached variables
    oldX = _cache['oldX']
    iK_cached = _cache['iK']
    if (oldX is None or X.size != oldX.size or iK_cached is None
            or n != _cache['oldn'] or np.any(X != oldX)):
        _cache['oldX'] = X.copy()
        _cache['oldn'] = n
        K = np.zeros((n, n, E))
        iK = np.zeros((n, n, E))
        beta = np.zeros((n, E))

        for i in range(E):
            inp = inputs / np.exp(X[:D, i])
            K[:, :, i] = np.exp(2 * X[D, i] - maha(inp, inp) / 2)
            if _has_field_gp0(gpmodel, 'nigp'):
                L = np.linalg.cholesky(
                    K[:, :, i] + np.exp(2 * X[D + 1, i]) * np.eye(n)
                    + np.diag(_get_field_gp0(gpmodel, 'nigp')[:, i])
                )
            else:
                L = np.linalg.cholesky(
                    K[:, :, i] + np.exp(2 * X[D + 1, i]) * np.eye(n)
                )
            iK[:, :, i] = cho_solve((L, True), np.eye(n))
            beta[:, i] = cho_solve((L, True), targets[:, i])
        _cache['K'] = K
        _cache['iK'] = iK
        _cache['beta'] = beta

    iK = _cache['iK']
    beta = _cache['beta']

    k = np.zeros((n, E))
    M = np.zeros(E)
    V = np.zeros((D, E))
    S = np.zeros((E, E))

    inp = inputs - m.reshape(1, -1)

    # 2) compute predicted mean and inv(s) times input-output covariance
    for i in range(E):
        iL = np.diag(np.exp(-X[:D, i]))
        in_ = inp @ iL
        B = iL @ s @ iL + np.eye(D)

        t = np.linalg.solve(B, in_.T).T
        l = np.exp(-np.sum(in_ * t, axis=1) / 2)
        lb = l * beta[:, i]
        tiL = t @ iL
        c = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(B))

        M[i] = np.sum(lb) * c
        V[:, i] = tiL.T @ lb * c
        k[:, i] = 2 * X[D, i] - np.sum(in_ * in_, axis=1) / 2

    # 3) compute predictive covariance, non-central moments
    for i in range(E):
        ii = inp / np.exp(2 * X[:D, i])

        for j in range(i + 1):
            R = s @ np.diag(np.exp(-2 * X[:D, i]) + np.exp(-2 * X[:D, j])) + np.eye(D)
            t = 1.0 / np.sqrt(np.linalg.det(R))
            ij = inp / np.exp(2 * X[:D, j])
            L = np.exp(
                k[:, i][:, np.newaxis] + k[:, j][np.newaxis, :]
                + maha(ii, -ij, np.linalg.solve(R, s) / 2)
            )
            if i == j:
                S[i, i] = t * (beta[:, i].T @ L @ beta[:, i] - np.sum(iK[:, :, i] * L))
            else:
                S[i, j] = beta[:, i].T @ L @ beta[:, j] * t
                S[j, i] = S[i, j]

        S[i, i] = S[i, i] + np.exp(2 * X[D, i])

    # 4) centralize moments
    S = S - np.outer(M, M)

    return M, S, V
