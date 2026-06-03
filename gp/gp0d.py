
"""
gp0d.py
*Summary:* Compute joint predictions and derivatives for multiple GPs
with uncertain inputs. Predictive variances contain uncertainty about the
function, but no noise.
If gpmodel.nigp exists, individial noise contributions are added.


  function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp0d(gpmodel, m, s)

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
  dMdm       output mean by input mean                             [ E  x  D ]
  dSdm       output covariance by input mean                       [E*E x  D ]
  dVdm       inv(s)*input-output covariance by input mean          [D*E x  D ]
  dMds       ouput mean by input covariance                        [ E  x D*D]
  dSds       output covariance by input covariance                 [E*E x D*D]
  dVds       inv(s)*input-output covariance by input covariance    [D*E x D*D]


Copyright (C) 2008-2013 by
Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.

Last modified: 2013-05-24

High-Level Steps
# If necessary, compute kernel matrix and cache it
# Compute predicted mean and inv(s) times input-output covariance
# Compute predictive covariance matrix, non-central moments
# Centralize moments
# Vectorize derivatives
"""

import numpy as np
from scipy.linalg import cho_solve

from pilco_python.util.maha import maha


_cache = {'K': None, 'iK': None, 'beta': None, 'oldX': None, 'oldn': None}


def _get_field_gp0d(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]


def _has_field_gp0d(obj, key):
    if hasattr(obj, 'get'):
        return key in obj
    return hasattr(obj, key)


def gp0d(gpmodel, m, s, compute_derivatives=False):
    global _cache

    inputs = _get_field_gp0d(gpmodel, 'inputs')
    targets = _get_field_gp0d(gpmodel, 'targets')
    X = _get_field_gp0d(gpmodel, 'hyp')
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
            if _has_field_gp0d(gpmodel, 'nigp'):
                L = np.linalg.cholesky(
                    K[:, :, i] + np.exp(2 * X[D + 1, i]) * np.eye(n)
                    + np.diag(_get_field_gp0d(gpmodel, 'nigp')[:, i])
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
    if compute_derivatives:
        dMds = np.zeros((E, D, D))
        dSdm = np.zeros((E, E, D))
        dSds = np.zeros((E, E, D, D))
        dVds = np.zeros((D, E, D, D))
        T = np.zeros((D, D))

    inp = inputs - m.reshape(1, -1)

    # 2) compute predicted mean and inv(s) times input-output covariance
    for i in range(E):
        iL = np.diag(np.exp(-X[:D, i]))
        in_ = inp @ iL
        B = iL @ s @ iL + np.eye(D)
        LiBL = iL @ np.linalg.solve(B, iL)
        t = np.linalg.solve(B, in_.T).T
        l = np.exp(-np.sum(in_ * t, axis=1) / 2)
        lb = l * beta[:, i]
        tL = t @ iL
        tlb = tL * lb[:, np.newaxis]
        c = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(B))
        M[i] = c * np.sum(lb)
        V[:, i] = tL.T @ lb * c
        if compute_derivatives:
            dMds[i, :, :] = c * tL.T @ tlb / 2 - LiBL * M[i] / 2
            for d in range(D):
                dVds[d, i, :, :] = (
                    c * (tL * tL[:, d][:, np.newaxis]).T @ tlb / 2
                    - LiBL * V[d, i] / 2
                    - (np.outer(V[:, i], LiBL[d, :]) + np.outer(LiBL[:, d], V[:, i])) / 2
                )
        k[:, i] = 2 * X[D, i] - np.sum(in_ * in_, axis=1) / 2
    if compute_derivatives:
        dMdm = V.T
        dVdm = 2 * np.transpose(dMds, (1, 0, 2))

    iell2 = np.exp(-2 * X[:D, :])  # D-by-E
    inpiell2 = inp[:, :, np.newaxis] * iell2[np.newaxis, :, :]

    # 3) compute predictive covariance matrix, non-central moments
    for i in range(E):
        ii = inpiell2[:, :, i]

        for j in range(i + 1):
            R = s @ np.diag(iell2[:, i] + iell2[:, j]) + np.eye(D)
            t_val = 1.0 / np.sqrt(np.linalg.det(R))
            ij_mat = inpiell2[:, :, j]
            L = np.exp(
                k[:, i][:, np.newaxis] + k[:, j][np.newaxis, :]
                + maha(ii, -ij_mat, np.linalg.solve(R, s) / 2)
            )
            if i == j:
                s1iKL = np.sum(iK[:, :, i] * L, axis=0, keepdims=True)
                s2iKL = np.sum(iK[:, :, i] * L, axis=1, keepdims=True)
                S[i, j] = t_val * (
                    beta[:, i].T @ L @ beta[:, i] - np.sum(s1iKL)
                )
                if compute_derivatives:
                    zi = np.linalg.solve(R.T, ii.T).T
                    bibLi = (L.T @ beta[:, i]) * beta[:, i]
                    cbLi = L.T @ (beta[:, i][:, np.newaxis] * zi)
                    r = (bibLi @ zi * 2 - (s2iKL.T + s1iKL) @ zi) * t_val
                    r = r.ravel()
                    for d in range(D):
                        T[d, :d + 1] = 2 * (
                            zi[:, :d + 1].T @ (zi[:, d] * bibLi)
                            + cbLi[:, :d + 1].T @ (zi[:, d] * beta[:, i])
                            - zi[:, :d + 1].T @ (zi[:, d] * s2iKL.ravel())
                            - zi[:, :d + 1].T @ (iK[:, :, i] * L @ zi[:, d])
                        )
                        T[:d + 1, d] = T[d, :d + 1]
            else:
                S[i, j] = beta[:, i].T @ L @ beta[:, j] * t_val
                S[j, i] = S[i, j]
                if compute_derivatives:
                    zi = np.linalg.solve(R.T, ii.T).T
                    zj = np.linalg.solve(R.T, ij_mat.T).T

                    bibLj = (L @ beta[:, j]) * beta[:, i]
                    bjbLi = (L.T @ beta[:, i]) * beta[:, j]
                    cbLi = L.T @ (beta[:, i][:, np.newaxis] * zi)
                    cbLj = L @ (beta[:, j][:, np.newaxis] * zj)

                    r = (bibLj @ zi + bjbLi @ zj) * t_val
                    r = r.ravel()
                    for d in range(D):
                        T[d, :d + 1] = (
                            zi[:, :d + 1].T @ (zi[:, d] * bibLj)
                            + cbLi[:, :d + 1].T @ (zj[:, d] * beta[:, j])
                            + zj[:, :d + 1].T @ (zj[:, d] * bjbLi)
                            + cbLj[:, :d + 1].T @ (zi[:, d] * beta[:, i])
                        )
                        T[:d + 1, d] = T[d, :d + 1]

            if compute_derivatives:
                dSdm[i, j, :] = r - M[i] * dMdm[j, :] - M[j] * dMdm[i, :]
                dSdm[j, i, :] = dSdm[i, j, :]
                T = (t_val * T - S[i, j] * np.diag(iell2[:, i] + iell2[:, j]) @ np.linalg.solve(R, np.eye(D))) / 2
                T = T - (M[i] * dMds[j, :, :] + M[j] * dMds[i, :, :])
                dSds[i, j, :, :] = T
                dSds[j, i, :, :] = T

        S[i, i] = S[i, i] + np.exp(2 * X[D, i])

    # 4) centralize moments
    S = S - np.outer(M, M)

    if not compute_derivatives:
        return M, S, V

    # 5) vectorize derivatives
    dMds = dMds.reshape(E, D * D, order='F')
    dSds = dSds.reshape(E * E, D * D, order='F')
    dSdm = dSdm.reshape(E * E, D, order='F')
    dVds = dVds.reshape(D * E, D * D, order='F')
    dVdm = dVdm.reshape(D * E, D, order='F')

    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds
