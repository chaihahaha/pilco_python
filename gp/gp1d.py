# gp1d.py
# *Summary:* Compute joint predictions (and derivatives) for the FITC sparse
# approximation to multiple GPs with uncertain inputs.
# Predictive variances contain uncertainty about the function, but no noise.
# If gpmodel.nigp exists, individual noise contributions are added.
#
#   function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp1d(gpmodel, m, s)
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
#   dMdm       output mean by input mean                             [ E  x  D ]
#   dSdm       output covariance by input mean                       [E*E x  D ]
#   dVdm       inv(s)*input-output covariance by input mean          [D*E x  D ]
#   dMds       output mean by input covariance                       [ E  x D*D]
#   dSds       output covariance by input covariance                 [E*E x D*D]
#   dVds       inv(s)*input-output covariance by input covariance    [D*E x D*D]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# High-Level Steps
# # If necessary, compute kernel matrix and cache it
# # Compute predicted mean and inv(s) times input-output covariance
# # Compute predictive covariance matrix, non-central moments
# # Centralize moments
# # Vectorize derivatives

import numpy as np
from scipy.linalg import cho_solve
from ..util.maha import maha

# Module-level cache (replaces MATLAB persistent iK iK2 beta oldX)
_cache = {
    'iK': None, 'iK2': None, 'beta': None, 'oldX': None
}

def _get_gpmodel_field(gpmodel, key):
    if hasattr(gpmodel, key):
        return getattr(gpmodel, key)
    return gpmodel[key]


def _has_gpmodel_field(gpmodel, key):
    if hasattr(gpmodel, 'get'):
        return key in gpmodel
    return hasattr(gpmodel, key)


def gp1d(gpmodel, m, s, compute_derivatives=False):
    # If no inducing inputs, back off to gp0d (no sparse GP required)
    if not _has_gpmodel_field(gpmodel, 'induce') or _get_gpmodel_field(gpmodel, 'induce').size == 0:
        from .gp0d import gp0d
        M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds = gp0d(gpmodel, m, s, compute_derivatives=True)
        M = M.reshape(-1, 1)
        return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds

    ridge = 1e-6                    # jitter to make matrix better conditioned
    n, D = _get_gpmodel_field(gpmodel, 'inputs').shape    # number of examples and dimension of inputs
    E = _get_gpmodel_field(gpmodel, 'targets').shape[1]    # number of examples and number of outputs
    X = _get_gpmodel_field(gpmodel, 'hyp')
    inp_data = _get_gpmodel_field(gpmodel, 'inputs')
    targets = _get_gpmodel_field(gpmodel, 'targets')

    np_count = _get_gpmodel_field(gpmodel, 'induce').shape[0]     # number of pseudo inputs per dimension
    pinput = _get_gpmodel_field(gpmodel, 'induce')                # all pseudo inputs
    if pinput.ndim == 2:
        pinput = pinput[:, :, np.newaxis]  # reshape to [np, D, 1]
    pE = pinput.shape[2]                   # third dimension of pseudo inputs

    # 1) If necessary: re-compute cached variables
    recalc = False
    if _cache['iK'] is None or _cache['iK2'] is None:
        recalc = True
    elif _cache['oldX'] is None or X.size != _cache['oldX'].size:
        recalc = True
    elif not np.array_equal(X, _cache['oldX']):
        recalc = True
    elif _cache['iK2'].size != E * np_count ** 2:
        recalc = True
    elif _cache['iK'].size != n * np_count * E:
        recalc = True

    if recalc:
        _cache['oldX'] = X.copy()                     # compute K, inv(K), inv(K2)
        _cache['iK'] = np.zeros((np_count, n, E))
        _cache['iK2'] = np.zeros((np_count, np_count, E))
        _cache['beta'] = np.zeros((np_count, E))

        for i in range(E):
            idx = min(i, pE - 1)
            pinp = pinput[:, :, idx] / np.exp(X[:D, i])           # [np, D]
            inp_scaled = inp_data / np.exp(X[:D, i])               # [n, D]
            Kmm = np.exp(2 * X[D, i] - maha(pinp, pinp) / 2) + ridge * np.eye(np_count)
            Kmn = np.exp(2 * X[D, i] - maha(pinp, inp_scaled) / 2)  # [np, n]
            L = np.linalg.cholesky(Kmm)                            # lower triangular
            V_int = np.linalg.solve(L, Kmn)                        # inv(sqrt(Kmm))*Kmn, [np, n]
            if _has_gpmodel_field(gpmodel, 'nigp'):
                G = np.exp(2 * X[D, i]) - np.sum(V_int ** 2, axis=0) + _get_gpmodel_field(gpmodel, 'nigp')[:, i]
            else:
                G = np.exp(2 * X[D, i]) - np.sum(V_int ** 2, axis=0)
            G = np.sqrt(1 + G / np.exp(2 * X[D+1, i]))             # [n]
            V_int = V_int / G                                      # bsxfun rdivide, [np, n]
            Am = np.linalg.cholesky(np.exp(2 * X[D+1, i]) * np.eye(np_count) + V_int @ V_int.T)
            At = L @ Am                                            # chol(sig*B), [np, np]
            iAt = np.linalg.solve(At, np.eye(np_count))            # [np, np]
            # The following is not an inverse matrix, but we'll treat it as such: multiply
            # the targets from right and the cross-covariances left to get predictive mean.
            temp = np.linalg.solve(Am, V_int / G)                  # [np, n]
            _cache['iK'][:, :, i] = (temp.T @ iAt).T               # [np, n]
            _cache['beta'][:, i] = _cache['iK'][:, :, i] @ targets[:, i]  # [np]
            iB = (iAt.T @ iAt) * np.exp(2 * X[D+1, i])             # inv(B)
            _cache['iK2'][:, :, i] = np.linalg.solve(Kmm, np.eye(np_count)) - iB

    iK_cache = _cache['iK']
    iK2_cache = _cache['iK2']
    beta = _cache['beta']

    k_mat = np.zeros((np_count, E))
    M = np.zeros((E, 1))
    V = np.zeros((D, E))
    S = np.zeros((E, E))
    dMds = np.zeros((E, D, D))
    dSdm = np.zeros((E, E, D))
    r = np.zeros(D)
    dSds = np.zeros((E, E, D, D))
    dVds = np.zeros((D, E, D, D))
    T = np.zeros((D, D))
    inp_arr = np.zeros((np_count, D, E))

    # 2) Compute predicted mean and inv(s) times input-output covariance
    for i in range(E):
        idx = min(i, pE - 1)
        inp_arr[:, :, i] = pinput[:, :, idx] - m.ravel()        # centralize p-inputs

        L_diag = np.diag(np.exp(-X[:D, i]))
        inn = inp_arr[:, :, i] @ L_diag                          # [np, D]
        B_mat = L_diag @ s @ L_diag + np.eye(D)
        LiBL = L_diag @ np.linalg.solve(B_mat, L_diag)           # iR

        t = np.linalg.solve(B_mat.T, inn.T).T                    # in/B, [np, D]
        l = np.exp(-np.sum(inn * t, axis=1) / 2)                 # [np]
        lb = l * beta[:, i]                                       # [np]
        tL = t @ L_diag                                           # [np, D]
        tlb = tL * lb[:, None]                                    # [np, D]
        c = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(B_mat))
        M[i] = c * np.sum(lb)
        V[:, i] = (tL.T @ lb) * c                                # inv(s) times input-output covariance
        dMds[i, :, :] = c * (tL.T @ tlb) / 2 - LiBL * M[i] / 2
        for d in range(D):
            dVds[d, i, :, :] = (c * (tL * tL[:, d, None]).T @ tlb / 2
                                - LiBL * V[d, i] / 2
                                - (V[:, i, None] @ LiBL[d:d+1, :]
                                   + LiBL[:, d:d+1] @ V[:, i, None].T) / 2)
        k_mat[:, i] = 2 * X[D, i] - np.sum(inn * inn, axis=1) / 2

    dMdm = V.T                                                  # derivatives wrt m, [E, D]
    dVdm = 2 * dMds.transpose(1, 0, 2)                          # [D, E, D] -> after reshape [D*E, D]

    iell2 = np.exp(-2 * X[:D, :])                               # [D, E]
    inpiell2 = inp_arr * iell2.reshape(1, D, E)               # N-by-D-by-E

    # 3) Compute predictive covariance matrix, non-central moments
    for i in range(E):
        ii = inpiell2[:, :, i]                                   # [np, D]

        for j in range(i + 1):
            R_mat = s @ np.diag(iell2[:, i] + iell2[:, j]) + np.eye(D)
            t_val = 1 / np.sqrt(np.linalg.det(R_mat))
            ij = inpiell2[:, :, j]                                # [np, D]
            L_mat = np.exp(k_mat[:, i, None] + k_mat[:, j][None, :]
                           + maha(ii, -ij, np.linalg.solve(R_mat, s) / 2))

            if i == j:
                iKL = iK2_cache[:, :, i] * L_mat                 # [np, np]
                s1iKL = np.sum(iKL, axis=0)                      # [np]
                s2iKL = np.sum(iKL, axis=1)                      # [np]
                S[i, j] = t_val * (beta[:, i].T @ L_mat @ beta[:, i] - np.sum(s1iKL))
                zi = np.linalg.solve(R_mat.T, ii.T).T            # ii/R, [np, D]
                bibLi = L_mat.T @ beta[:, i] * beta[:, i]        # [np]
                cbLi = L_mat.T @ (beta[:, i, None] * zi)         # [np, D]
                r = (bibLi.T @ zi * 2 - (s2iKL[None, :] + s1iKL[None, :]) @ zi) * t_val  # [D]
                for d in range(D):
                    idx_vec = np.arange(d + 1)
                    T[d, :d+1] = 2 * (zi[:, idx_vec].T @ (zi[:, d] * bibLi)
                                      + cbLi[:, idx_vec].T @ (zi[:, d] * beta[:, i])
                                      - zi[:, idx_vec].T @ (zi[:, d] * s2iKL)
                                      - zi[:, idx_vec].T @ (iKL @ zi[:, d]))
                    T[:d+1, d] = T[d, :d+1]
            else:
                zi = np.linalg.solve(R_mat.T, ii.T).T
                zj = np.linalg.solve(R_mat.T, ij.T).T
                S[i, j] = beta[:, i].T @ L_mat @ beta[:, j] * t_val
                S[j, i] = S[i, j]

                bibLj = L_mat @ beta[:, j] * beta[:, i]           # [np]
                bjbLi = L_mat.T @ beta[:, i] * beta[:, j]         # [np]
                cbLi = L_mat.T @ (beta[:, i, None] * zi)          # [np, D]
                cbLj = L_mat @ (beta[:, j, None] * zj)            # [np, D]

                r = (bibLj.T @ zi + bjbLi.T @ zj) * t_val          # [D]
                for d in range(D):
                    idx_vec = np.arange(d + 1)
                    T[d, :d+1] = (zi[:, idx_vec].T @ (zi[:, d] * bibLj)
                                  + cbLi[:, idx_vec].T @ (zj[:, d] * beta[:, j])
                                  + zj[:, idx_vec].T @ (zj[:, d] * bjbLi)
                                  + cbLj[:, idx_vec].T @ (zi[:, d] * beta[:, i]))
                    T[:d+1, d] = T[d, :d+1]

            dSdm[i, j, :] = r - M[i] * dMdm[j, :] - M[j] * dMdm[i, :]
            dSdm[j, i, :] = dSdm[i, j, :]
            T2 = (t_val * T - S[i, j] * np.diag(iell2[:, i] + iell2[:, j]) @ np.linalg.inv(R_mat)) / 2
            T2 = T2 - (M[i] * dMds[j, :, :] + M[j] * dMds[i, :, :]).reshape(D, D)
            dSds[i, j, :, :] = T2
            dSds[j, i, :, :] = T2

        S[i, i] = S[i, i] + np.exp(2 * X[D, i])

    # 4) Centralize moments
    S = S - M @ M.T

    # 5) Vectorize derivatives (Fortran order to match MATLAB column-major)
    dMds = dMds.reshape(E, D * D, order='F')
    dSds = dSds.reshape(E * E, D * D, order='F')
    dSdm = dSdm.reshape(E * E, D, order='F')
    dVds = dVds.reshape(D * E, D * D, order='F')
    # dVdm is from transpose; make C-contiguous before Fortran reshape
    dVdm = np.ascontiguousarray(dVdm).reshape(D * E, D, order='F')

    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds
