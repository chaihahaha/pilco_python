
"""
gp1.py
*Summary:* Compute joint predictions for the FITC sparse approximation to
multiple GPs with uncertain inputs.
Predictive variances contain uncertainty about the function, but no noise.
If gpmodel.nigp exists, individual noise contributions are added.

  function [M, S, V] = gp1d(gpmodel, m, s)

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

Last modified: 2013-03-05

High-Level Steps
# If necessary, compute kernel matrix and cache it
# Compute predicted mean and inv(s) times input-output covariance
# Compute predictive covariance matrix, non-central moments
# Centralize moments
"""

import numpy as np
from scipy.linalg import solve_triangular

from pilco_python.util.maha import maha
from .gp0 import gp0


_cache = {'iK': None, 'iK2': None, 'beta': None, 'oldX': None, 'oldniK': 0, 'oldniK2': 0}


def _get_field_gp1(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]


def _has_field_gp1(obj, key):
    if hasattr(obj, 'get'):
        return key in obj
    return hasattr(obj, key)


def gp1(gpmodel, m, s):
    global _cache

    if not _has_field_gp1(gpmodel, 'induce') or _get_field_gp1(gpmodel, 'induce').size == 0:
        return gp0(gpmodel, m, s)

    ridge = 1e-6
    inputs = _get_field_gp1(gpmodel, 'inputs')
    targets = _get_field_gp1(gpmodel, 'targets')
    X = _get_field_gp1(gpmodel, 'hyp')
    n, D = inputs.shape
    n2, E = targets.shape
    if n != n2:
        raise ValueError('inputs and targets must have same number of rows')

    pinput = _get_field_gp1(gpmodel, 'induce')
    np_val, pD, pE = pinput.shape

    # 1) If necessary: re-compute cached variables
    oldX = _cache['oldX']
    iK_cached = _cache['iK']
    iK2_cached = _cache['iK2']
    niK = _cache['oldniK']
    niK2 = _cache['oldniK2']
    if (oldX is None or X.size != oldX.size or iK_cached is None
            or iK2_cached is None or np.any(X != oldX)
            or niK2 != E * np_val**2 or niK != n * np_val * E):
        _cache['oldX'] = X.copy()
        _cache['oldniK'] = n * np_val * E
        _cache['oldniK2'] = E * np_val**2
        iK = np.zeros((np_val, n, E))
        iK2 = np.zeros((np_val, np_val, E))
        beta = np.zeros((np_val, E))

        for i in range(E):
            pinp = pinput[:, :, min(i, pE - 1)] / np.exp(X[:D, i])
            inp = inputs / np.exp(X[:D, i])
            Kmm = np.exp(2 * X[D, i] - maha(pinp, pinp) / 2) + ridge * np.eye(np_val)
            Kmn = np.exp(2 * X[D, i] - maha(pinp, inp) / 2)
            L = np.linalg.cholesky(Kmm)
            V = solve_triangular(L, Kmn, lower=True)
            if _has_field_gp1(gpmodel, 'nigp'):
                G = np.exp(2 * X[D, i]) - np.sum(V**2, axis=0) + _get_field_gp1(gpmodel, 'nigp')[:, i]
            else:
                G = np.exp(2 * X[D, i]) - np.sum(V**2, axis=0)
            G = np.sqrt(1 + G / np.exp(2 * X[D + 1, i]))
            V = V / G
            Am = np.linalg.cholesky(np.exp(2 * X[D + 1, i]) * np.eye(np_val) + V @ V.T)
            At = L @ Am
            iAt = np.linalg.solve(At, np.eye(np_val))
            Xtmp = solve_triangular(Am, V / G, lower=True)
            iK[:, :, i] = (Xtmp.T @ iAt).T
            beta[:, i] = iK[:, :, i] @ targets[:, i]
            iB = (iAt.T @ iAt) * np.exp(2 * X[D + 1, i])
            iK2[:, :, i] = np.linalg.solve(Kmm, np.eye(np_val)) - iB

        _cache['iK'] = iK
        _cache['iK2'] = iK2
        _cache['beta'] = beta

    iK = _cache['iK']
    iK2 = _cache['iK2']
    beta = _cache['beta']

    k = np.zeros((np_val, E))
    M = np.zeros(E)
    V = np.zeros((D, E))
    S = np.zeros((E, E))
    inp = np.zeros((np_val, D, E))

    # 2) Compute predicted mean and inv(s) times input-output covariance
    for i in range(E):
        inp[:, :, i] = pinput[:, :, min(i, pE - 1)] - m.reshape(1, -1)

        L_diag = np.diag(np.exp(-X[:D, i]))
        in_ = inp[:, :, i] @ L_diag
        B = L_diag @ s @ L_diag + np.eye(D)

        t = np.linalg.solve(B, in_.T).T
        l = np.exp(-np.sum(in_ * t, axis=1) / 2)
        lb = l * beta[:, i]
        tL = t @ L_diag
        c = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(B))

        M[i] = np.sum(lb) * c
        V[:, i] = tL.T @ lb * c
        k[:, i] = 2 * X[D, i] - np.sum(in_ * in_, axis=1) / 2

    # 3) Compute predictive covariance matrix, non-central moments
    for i in range(E):
        ii = inp[:, :, i] / np.exp(2 * X[:D, i])

        for j in range(i + 1):
            R = s @ np.diag(np.exp(-2 * X[:D, i]) + np.exp(-2 * X[:D, j])) + np.eye(D)
            t = 1.0 / np.sqrt(np.linalg.det(R))
            ij = inp[:, :, j] / np.exp(2 * X[:D, j])
            L = np.exp(
                k[:, i][:, np.newaxis] + k[:, j][np.newaxis, :]
                + maha(ii, -ij, np.linalg.solve(R, s) / 2)
            )
            if i == j:
                S[i, i] = t * (beta[:, i].T @ L @ beta[:, i] - np.sum(iK2[:, :, i] * L))
            else:
                S[i, j] = beta[:, i].T @ L @ beta[:, j] * t
                S[j, i] = S[i, j]

        S[i, i] = S[i, i] + np.exp(2 * X[D, i])

    # 4) Centralize moments
    S = S - np.outer(M, M)

    return M, S, V
