# gp2.py
# *Summary:* Compute joint predictions and derivatives for multiple GPs
# with uncertain inputs. Does not consider the uncertainty about the underlying
# function (in prediction), hence, only the GP mean function is considered.
# Therefore, this representation is equivalent to a regularized RBF network.
# If gpmodel.nigp exists, individual noise contributions are added.
#
#   function [M, S, V] = gp2(gpmodel, m, s)
#
# *Input arguments:*
#
#   gpmodel    GP model struct
#     hyp      log-hyper-parameters                                  [D+2 x  E ]
#     inputs   training inputs                                       [ n  x  D ]
#     targets  training targets                                      [ n  x  E ]
#     nigp     (optional) individual noise variance terms            [ n  x  E ]
#   m          mean of the test distribution                         [ D  x  1 ]
#   s          covariance matrix of the test distribution            [ D  x  D ]
#
# *Output arguments:*
#
#   M          mean of pred. distribution                            [ E  x  1 ]
#   S          covariance of the pred. distribution                  [ E  x  E ]
#   V          inv(s) times covariance between input and output      [ D  x  E ]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# High-Level Steps
# # If necessary, re-compute cached variables
# # Compute predicted mean and inv(s) times input-output covariance
# # Compute predictive covariance matrix, non-central moments
# # Centralize moments

import numpy as np
from scipy.linalg import cho_solve
from ..util.maha import maha

# Module-level cache (replaces MATLAB persistent)
_cache = {
    'iK': None, 'oldX': None, 'oldIn': None, 'oldOut': None,
    'beta': None, 'oldn': None
}

def _get_field(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]


def _has_field(obj, key):
    if hasattr(obj, 'get'):
        return key in obj
    return hasattr(obj, key)


def gp2(gpmodel, m, s):
    D = _get_field(gpmodel, 'inputs').shape[1]     # number of examples and dimension of inputs
    n, E = _get_field(gpmodel, 'targets').shape    # number of examples and number of outputs

    inp = _get_field(gpmodel, 'inputs')
    target = _get_field(gpmodel, 'targets')
    X = _get_field(gpmodel, 'hyp')

    # 1) if necessary: re-compute cached variables
    recalc = False
    if _cache['oldX'] is None or _cache['iK'] is None:
        recalc = True
    elif X.size != _cache['oldX'].size:
        recalc = True
    elif n != _cache['oldn']:
        recalc = True
    elif not np.array_equal(X, _cache['oldX']):
        recalc = True
    elif not np.array_equal(inp, _cache['oldIn']):
        recalc = True
    elif not np.array_equal(target, _cache['oldOut']):
        recalc = True

    if recalc:
        _cache['oldX'] = X.copy()
        _cache['oldIn'] = inp.copy()
        _cache['oldOut'] = target.copy()
        _cache['oldn'] = n
        _cache['iK'] = np.zeros((n, n, E))
        _cache['beta'] = np.zeros((n, E))

        for i in range(E):                                       # compute K and inv(K)
            inp_scaled = inp / np.exp(X[:D, i])                  # [n, D] / [D] -> [n, D]
            K_i = np.exp(2 * X[D, i] - maha(inp_scaled, inp_scaled) / 2)
            if _has_field(gpmodel, 'nigp'):
                K_noisy = K_i + np.exp(2 * X[D+1, i]) * np.eye(n) + np.diag(_get_field(gpmodel, 'nigp')[:, i])
            else:
                K_noisy = K_i + np.exp(2 * X[D+1, i]) * np.eye(n)
            L = np.linalg.cholesky(K_noisy)                      # lower triangular
            _cache['iK'][:, :, i] = cho_solve((L, True), np.eye(n))
            _cache['beta'][:, i] = cho_solve((L, True), target[:, i])

    iK_cache = _cache['iK']
    beta = _cache['beta']

    k_mat = np.zeros((n, E))
    M = np.zeros((E, 1))
    V = np.zeros((D, E))
    S = np.zeros((E, E))

    # centralize inputs
    inp_c = inp - m.ravel()                                     # [n, D] - [D] -> [n, D]

    # 2) Compute predicted mean and inv(s) times input-output covariance
    for i in range(E):
        iL = np.diag(np.exp(-X[:D, i]))                          # inverse length-scales, [D, D]
        inn = inp_c @ iL                                         # [n, D]
        B = iL @ s @ iL + np.eye(D)

        t = np.linalg.solve(B.T, inn.T).T                        # in/B -> [n, D]
        l = np.exp(-np.sum(inn * t, axis=1) / 2)                 # [n]
        lb = l * beta[:, i]                                      # [n]
        tL = t @ iL                                              # [n, D]
        c = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(B))

        M[i] = np.sum(lb) * c                                    # predicted mean
        V[:, i] = tL.T @ lb * c                                  # inv(s) times input-output covariance
        k_mat[:, i] = 2 * X[D, i] - np.sum(inn * inn, axis=1) / 2

    # 3) Compute predictive covariance, non-central moments
    for i in range(E):
        ii = inp_c / np.exp(2 * X[:D, i])

        for j in range(i + 1):
            R = s @ np.diag(np.exp(-2 * X[:D, i]) + np.exp(-2 * X[:D, j])) + np.eye(D)
            t_val = 1 / np.sqrt(np.linalg.det(R))
            ij = inp_c / np.exp(2 * X[:D, j])
            L_mat = np.exp(k_mat[:, i, None] + k_mat[:, j][None, :] + maha(ii, -ij, np.linalg.solve(R, s) / 2))
            S[i, j] = t_val * beta[:, i].T @ L_mat @ beta[:, j]
            S[j, i] = S[i, j]

        S[i, i] = S[i, i] + 1e-6         # add small jitter for numerical reasons

    # 4) Centralize moments
    S = S - M @ M.T

    return M, S, V
