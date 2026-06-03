# gp2d.py
# *Summary:* Compute joint predictions and derivatives for multiple GPs
# with uncertain inputs. Does not consider the uncertainty about the underlying
# function (in prediction), hence, only the GP mean function is considered.
# Therefore, this representation is equivalent to a regularized RBF network.
# If gpmodel.nigp exists, individual noise contributions are added.
#
#   function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdi, dSdi, dVdi, ...
#                      dMdt, dSdt, dVdt, dMdX, dSdX, dVdX] = gp2d(gpmodel, m, s)
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
#   dMds       output mean by input covariance                        [ E  x D*D]
#   dSds       output covariance by input covariance                 [E*E x D*D]
#   dVds       inv(s)*input-output covariance by input covariance    [D*E x D*D]
#
#   dMdi      output mean by inputs                                  [ E  x n*D]
#   dSdi      output covariance by inputs                            [E*E x n*D]
#   dVdi      inv(s) times input-output covariance by inputs         [D*E x n*D]
#   dMdt      output mean by targets                                 [ E  x n*E]
#   dSdt      output covariance by targets                           [E*E x n*E]
#   dVdt      inv(s) times input-output covariance by targets        [D*E x n*E]
#   dMdX      output mean by hyperparameters                         [ E  x P*E]
#   dSdX      output covariance by hyperparameters                   [E*E x P*E]
#   dVdX      inv(s) times input-output covariance by hyper-par.     [D*E x P*E]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# High-Level Steps
# # If necessary, re-compute cached variables
# # Compute predicted mean and inv(s) times input-output covariance
# # Compute predictive covariance matrix, non-central moments
# # Centralize moments
# # Vectorize derivatives

import numpy as np
from scipy.linalg import cho_solve
from ..util.maha import maha

# Module-level cache (replaces MATLAB persistent K iK oldX oldIn oldOut beta oldn)
_cache = {
    'K': None, 'iK': None, 'oldX': None, 'oldIn': None,
    'oldOut': None, 'beta': None, 'oldn': None
}

def _get_field_2d(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]


def _has_field_2d(obj, key):
    if hasattr(obj, 'get'):
        return key in obj
    return hasattr(obj, key)


def gp2d(gpmodel, m, s):
    inp = _get_field_2d(gpmodel, 'inputs')
    target = _get_field_2d(gpmodel, 'targets')
    X_raw = _get_field_2d(gpmodel, 'hyp')

    D = inp.shape[1]                     # number of examples and dimension of input space
    n, E = target.shape                  # number of examples and number of outputs
    X = X_raw.reshape(D + 2, E)          # [D+2, E]

    # 1) If necessary, re-compute cached variables
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
        _cache['K'] = np.zeros((n, n, E))
        _cache['iK'] = np.zeros((n, n, E))
        _cache['beta'] = np.zeros((n, E))

        # compute K and inv(K) and beta
        for i in range(E):
            inp_scaled = inp / np.exp(X[:D, i])
            K_i = np.exp(2 * X[D, i] - maha(inp_scaled, inp_scaled) / 2)
            _cache['K'][:, :, i] = K_i
            if _has_field_2d(gpmodel, 'nigp'):
                K_noisy = K_i + np.exp(2 * X[D+1, i]) * np.eye(n) + np.diag(_get_field_2d(gpmodel, 'nigp')[:, i])
            else:
                K_noisy = K_i + np.exp(2 * X[D+1, i]) * np.eye(n)
            L = np.linalg.cholesky(K_noisy)
            _cache['iK'][:, :, i] = cho_solve((L, True), np.eye(n))
            _cache['beta'][:, i] = cho_solve((L, True), target[:, i])

    K_all = _cache['K']
    iK_all = _cache['iK']
    beta = _cache['beta']

    # initializations
    k_mat = np.zeros((n, E))
    M = np.zeros((E, 1))
    V = np.zeros((D, E))
    S = np.zeros((E, E))
    dMds = np.zeros((E, D, D))
    dSdm = np.zeros((E, E, D))
    r = np.zeros(D)
    dSds = np.zeros((E, E, D, D))
    dVds = np.zeros((D, E, D, D))
    T_mat = np.zeros((D, D))
    tlbdi = np.zeros((n, D))
    dMdi = np.zeros((E, n, D))
    dMdt = np.zeros((E, n, E))
    dVdt = np.zeros((D, E, n, E))
    dVdi = np.zeros((D, E, n, D))
    dSdt = np.zeros((E, E, n, E))
    dSdi = np.zeros((E, E, n, D))
    dMdX = np.zeros((E, D + 2, E))
    dSdX = np.zeros((E, E, D + 2, E))
    dVdX = np.zeros((D, E, D + 2, E))
    Z_mat = np.zeros((n, D))
    bdX = np.zeros((n, E, D))
    kdX = np.zeros((n, E, D + 1))

    # centralize training inputs
    inp_c = inp - m.ravel()                                     # [n, D]

    # 2) compute predicted mean and input-output covariance
    for i in range(E):
        # first some useful intermediate terms
        K2 = K_all[:, :, i] + np.exp(2 * X[D+1, i]) * np.eye(n)  # K + sigma^2*I
        inp2 = inp / np.exp(X[:D, i])                             # [n, D]
        ii = inp / np.exp(2 * X[:D, i])                           # [n, D]
        R_vec = s + np.diag(np.exp(2 * X[:D, i]))
        L_diag = np.diag(np.exp(-X[:D, i]))
        B_mat = L_diag @ s @ L_diag + np.eye(D)
        iR = L_diag @ np.linalg.solve(B_mat, L_diag)            # L/B*L = inv(R)
        t = inp_c @ iR                                            # inp*iR, [n, D]
        l = np.exp(-np.sum(t * inp_c, axis=1) / 2)                # [n]
        lb = l * beta[:, i]                                       # [n]
        tliK = t.T @ (l[:, None] * iK_all[:, :, i])               # [D, n]
        liK = np.linalg.solve(K2, l)                              # K2\l, [n]
        tlb = t * lb[:, None]                                     # [n, D]

        c_val = np.exp(2 * X[D, i]) / np.sqrt(np.linalg.det(R_vec)) * np.exp(np.sum(X[:D, i]))
        detR = np.linalg.det(R_vec)
        diag_iR = np.diag(iR)                                     # [D]
        detdX_vec = detR * diag_iR * (2 * np.exp(2 * X[:D, i]))  # d(det R)/dX, [D]
        cdX = -0.5 * c_val / detR * detdX_vec + c_val             # derivs w.r.t length-scales, [D]
        dldX = l[:, None] * (t * (2 * np.exp(2 * X[:D, i])) * t / 2)  # [n, D]

        M[i] = np.sum(lb) * c_val                                 # predicted mean

        iK2beta = np.linalg.solve(K2, beta[:, i])                 # K2\beta
        dMds[i, :, :] = c_val * (t.T @ tlb) / 2 - iR * M[i] / 2
        dMdX[i, D + 1, i] = -c_val * np.sum(l * (2 * np.exp(2 * X[D+1, i]) * iK2beta))
        dMdX[i, D, i] = -dMdX[i, D + 1, i]

        dVdX[:, i, D + 1, i] = -((l * (2 * np.exp(2 * X[D+1, i]) * iK2beta)) @ t * c_val)
        dVdX[:, i, D, i] = -dVdX[:, i, D + 1, i]

        dsi = -inp2 * 2 * inp2                                    # d(sum(inp2.*inp2,2))/dX, [n, D]
        dslb = np.zeros(D)

        for d in range(D):
            sqdi = K_all[:, :, i] * (ii[:, d, None] - ii[:, d][None, :])    # [n, n]
            sqdiBi = sqdi @ beta[:, i]                                       # [n]
            tlbdi[:, d] = sqdi @ liK * beta[:, i] + sqdiBi * liK            # [n]
            tlbdi2 = -tliK @ (-(sqdi * beta[:, i, None]).T - np.diag(sqdiBi))  # [D, n]
            dVdi[:, i, :, d] = c_val * (iR[:, d][:, None] @ lb[None, :]
                                        - (t * tlb[:, d, None]).T + tlbdi2)
            dsqdX = (dsi[:, d, None] + dsi[:, d][None, :]
                     + 4 * inp2[:, d, None] * inp2[:, d][None, :])           # [n, n]
            dKdX = -K_all[:, :, i] * dsqdX / 2                                # dK/dX(1:D), [n, n]
            dKdXbeta = dKdX @ beta[:, i]                                      # [n]
            bdX[:, i, d] = -np.linalg.solve(K2, dKdXbeta)                     # dbeta/dX, [n]
            dslb[d] = -liK.T @ dKdXbeta + beta[:, i].T @ dldX[:, d]
            dlb = dldX[:, d] * beta[:, i] + l * bdX[:, i, d]                  # [n]
            dtdX = inp_c @ (-iR[:, d, None] * 2 * np.exp(2 * X[d, i]) * iR[d:d+1, :])
            dlbt = lb.T @ dtdX + dlb.T @ t                                    # [D]
            dVdX[:, i, d, i] = (dlbt * c_val + cdX[d] * (lb.T @ t))           # [D]

        # end d

        dMdi[i, :, :] = c_val * (tlbdi - tlb)
        dMdt[i, :, i] = c_val * liK                                       # [n] -> row [n]
        dMdX[i, :D, i] = cdX * np.sum(beta[:, i] * l) + c_val * dslb      # [D]
        v = inp_c / np.exp(X[:D, i])                                        # [n, D]
        k_mat[:, i] = 2 * X[D, i] - np.sum(v * v, axis=1) / 2
        V[:, i] = t.T @ lb * c_val                                        # input-output covariance

        for d in range(D):
            dVds[d, i, :, :] = (c_val * (t * t[:, d, None]).T @ tlb / 2
                                - iR * V[d, i] / 2
                                - V[:, i, None] @ iR[d:d+1, :] / 2
                                - iR[:, d:d+1] @ V[:, i, None].T / 2)
            kdX[:, i, d] = v[:, d] * v[:, d]

        # end d

        dVdt[:, i, :, i] = c_val * tliK
        kdX[:, i, D] = 2 * np.ones(n)                                    # pre-computation for later

    # end i
    dMdm = V.T                                                           # derivatives w.r.t m, [E, D]
    dVdm = 2 * dMds.transpose(1, 0, 2)                                   # [D, E, D]

    # 3) predictive covariance matrix (non-central moments)
    for i in range(E):
        K2_i = K_all[:, :, i] + np.exp(2 * X[D+1, i]) * np.eye(n)
        ii = inp_c / np.exp(2 * X[:D, i])

        for j in range(i + 1):  # if i==j: diagonal elements of S
            R_mat = s @ np.diag(np.exp(-2 * X[:D, i]) + np.exp(-2 * X[:D, j])) + np.eye(D)
            t_val = 1 / np.sqrt(np.linalg.det(R_mat))
            if np.linalg.cond(R_mat) > 1e15:
                import warnings
                warnings.warn('R-matrix in gp2d ill-conditioned')
            iR_mat = np.linalg.inv(R_mat)
            ij = inp_c / np.exp(2 * X[:D, j])
            L_mat = np.exp(k_mat[:, i, None] + k_mat[:, j][None, :]
                           + maha(ii, -ij, np.linalg.solve(R_mat, s) / 2))  # called Q in thesis
            A = beta[:, i, None] @ beta[:, j][None, :]                       # [n, n]
            A = A * L_mat
            ssA = np.sum(A)
            S[i, j] = t_val * ssA
            S[j, i] = S[i, j]

            zzi = ii @ (np.linalg.solve(R_mat, s))
            zzj = ij @ (np.linalg.solve(R_mat, s))
            zi = np.linalg.solve(R_mat.T, ii.T).T                             # ii/R
            zj = np.linalg.solve(R_mat.T, ij.T).T                             # ij/R

            tdX = -0.5 * t_val * np.sum(iR_mat.T * (s * (-2 * np.exp(-2 * X[:D, i])
                                                              - 2 * np.exp(-2 * X[:D, i]))), axis=0)
            tdXi = -0.5 * t_val * np.sum(iR_mat.T * (s * (-2 * np.exp(-2 * X[:D, i]))), axis=0)
            tdXj = -0.5 * t_val * np.sum(iR_mat.T * (s * (-2 * np.exp(-2 * X[:D, j]))), axis=0)
            bLiKi = iK_all[:, :, j] @ (L_mat.T @ beta[:, i])                   # [n]
            bLiKj = iK_all[:, :, i] @ (L_mat @ beta[:, j])                     # [n]

            Q2 = np.linalg.solve(R_mat, s) / 2
            aQ = ii @ Q2
            bQ = ij @ Q2

            for d in range(D):
                Z_mat[:, d] = (np.exp(-2 * X[d, i]) * (A @ zzj[:, d] + np.sum(A, axis=1) * (zzi[:, d] - inp_c[:, d]))
                               + np.exp(-2 * X[d, j]) * (zzi[:, d] @ A + np.sum(A, axis=0) * (zzj[:, d] - inp_c[:, d])))
                Q = (inp_c[:, d, None] - inp_c[:, d][None, :])                  # [n, n]
                B_mat2 = K_all[:, :, i] * Q
                Z_mat[:, d] = Z_mat[:, d] + np.exp(-2 * X[d, i]) * (B_mat2 @ beta[:, i] * bLiKj + beta[:, i] * (B_mat2 @ bLiKj))

                if i != j:
                    B_mat3 = K_all[:, :, j] * Q
                    B_j = B_mat3
                else:
                    B_j = B_mat2

                Z_mat[:, d] = Z_mat[:, d] + np.exp(-2 * X[d, j]) * (bLiKi * (B_j @ beta[:, j])
                                                                     + B_j @ bLiKi * beta[:, j])
                B = (zi[:, d, None] + zj[:, d][None, :]) * A
                r[d] = np.sum(B) * t_val
                idx_vec = np.arange(d + 1)
                T_mat[d, :d+1] = np.sum(zi[:, idx_vec].T @ B, axis=1) + np.sum(B @ zj[:, idx_vec], axis=0)
                T_mat[:d+1, d] = T_mat[d, :d+1]

                if i == j:
                    RTi = s * (-2 * np.exp(-2 * X[:D, i]) - 2 * np.exp(-2 * X[:D, j]))
                    diRi = -iR_mat @ (RTi[:, d:d+1] @ iR_mat[d:d+1, :])
                else:
                    RTi = s * (-2 * np.exp(-2 * X[:D, i]))
                    RTj = s * (-2 * np.exp(-2 * X[:D, j]))
                    diRi = -iR_mat @ (RTi[:, d:d+1] @ iR_mat[d:d+1, :])
                    diRj = -iR_mat @ (RTj[:, d:d+1] @ iR_mat[d:d+1, :])
                    QdXj = diRj @ s / 2                                  # dQ2/dXj

                QdXi = diRi @ s / 2                                      # dQ2/dXi

                if i == j:
                    daQi = ii @ QdXi + (-2 * ii[:, d, None] * Q2[:, d][None, :])
                    dsaQi = np.sum(daQi * ii, axis=1) - 2 * aQ[:, d] * ii[:, d]
                    dsaQj = dsaQi
                    dsbQi = dsaQi
                    dsbQj = dsbQi
                    dm2i = (-2 * daQi @ ii.T
                            + 2 * (aQ[:, d, None] @ ii[:, d][None, :]
                                   + ii[:, d, None] @ aQ[:, d][None, :]))
                    dm2j = dm2i
                else:
                    dbQi = ij @ QdXi
                    dbQj = ij @ QdXj + (-2 * ij[:, d, None] * Q2[:, d][None, :])
                    daQi = ii @ QdXi + (-2 * ii[:, d, None] * Q2[:, d][None, :])
                    daQj = ii @ QdXj

                    dsaQi = np.sum(daQi * ii, axis=1) - 2 * aQ[:, d] * ii[:, d]
                    dsaQj = np.sum(daQj * ii, axis=1)
                    dsbQi = np.sum(dbQi * ij, axis=1)
                    dsbQj = np.sum(dbQj * ij, axis=1) - 2 * bQ[:, d] * ij[:, d]
                    dm2i = -2 * daQi @ ij.T
                    dm2j = -2 * ii @ dbQj.T

                dm1i = dsaQi[:, None] + dsbQi[None, :]                    # first part of maha(..) wrt Xi
                dm1j = dsaQj[:, None] + dsbQj[None, :]                    # first part of maha(..) wrt Xj
                dmahai = dm1i - dm2i
                dmahaj = dm1j - dm2j

                if i == j:
                    LdXi = L_mat * (dmahai + kdX[:, i, d, None] + kdX[:, j, d][None, :])
                    dSdX[i, i, d, i] = beta[:, i].T @ LdXi @ beta[:, j]
                else:
                    LdXi = L_mat * (dmahai + kdX[:, i, d, None])
                    LdXj = L_mat * (dmahaj + kdX[:, j, d][None, :])
                    dSdX[i, j, d, i] = beta[:, i].T @ LdXi @ beta[:, j]
                    dSdX[i, j, d, j] = beta[:, i].T @ LdXj @ beta[:, j]

            # end d

            if i == j:
                dSdX[i, i, :D, i] = (dSdX[i, i, :D, i]
                                      + bdX[:, i, :].T @ (L_mat + L_mat.T) @ beta[:, i])
                dSdX[i, i, :D, i] = t_val * dSdX[i, i, :D, i] + tdX * ssA
                dSdX[i, i, D + 1, i] = (2 * np.exp(2 * X[D+1, i]) * t_val
                                         * (-np.sum(beta[:, i] * bLiKi) - np.sum(beta[:, i] * bLiKi)))
            else:
                dSdX[i, j, :D, i] = dSdX[i, j, :D, i] + bdX[:, i, :].T @ (L_mat @ beta[:, j])
                dSdX[i, j, :D, j] = dSdX[i, j, :D, j] + bdX[:, j, :].T @ (L_mat.T @ beta[:, i])
                dSdX[i, j, :D, i] = t_val * dSdX[i, j, :D, i] + tdXi * ssA
                dSdX[i, j, :D, j] = t_val * dSdX[i, j, :D, j] + tdXj * ssA
                dSdX[i, j, D + 1, i] = 2 * np.exp(2 * X[D+1, i]) * t_val * (-beta[:, i].T @ bLiKj)
                dSdX[i, j, D + 1, j] = 2 * np.exp(2 * X[D+1, j]) * t_val * (-beta[:, j].T @ bLiKi)

            dSdm[i, j, :] = r - M[i] * dMdm[j, :] - M[j] * dMdm[i, :]
            dSdm[j, i, :] = dSdm[i, j, :]
            T2 = (t_val * T_mat
                  - S[i, j] * np.diag(np.exp(-2 * X[:D, i]) + np.exp(-2 * X[:D, j])) @ iR_mat) / 2
            T2 = T2 - (M[i] * dMds[j, :, :] + M[j] * dMds[i, :, :]).reshape(D, D)
            dSds[i, j, :, :] = T2
            dSds[j, i, :, :] = T2

            if i == j:
                dSdt[i, i, :, i] = (np.linalg.solve(K2_i, beta[:, i].T @ (L_mat + L_mat.T)) * t_val
                                     - 2 * dMdt[i, :, i] * M[i])
                dSdX[i, j, :, i] = dSdX[i, j, :, i] - M[i] * dMdX[j, :, j] - M[j] * dMdX[i, :, i]
            else:
                K2_j = K_all[:, :, j] + np.exp(2 * X[D+1, j]) * np.eye(n)
                dSdt[i, j, :, i] = (np.linalg.solve(K2_i, beta[:, j].T @ L_mat.T) * t_val
                                    - dMdt[i, :, i] * M[j])
                dSdt[i, j, :, j] = (np.linalg.solve(K2_j, beta[:, i].T @ L_mat) * t_val
                                    - dMdt[j, :, j] * M[i])
                dSdt[j, i, :, :] = dSdt[i, j, :, :]
                dSdX[i, j, :, j] = dSdX[i, j, :, j] - M[i] * dMdX[j, :, j]
                dSdX[i, j, :, i] = dSdX[i, j, :, i] - M[j] * dMdX[i, :, i]

            dSdi[i, j, :, :] = Z_mat * t_val - (M[i] * dMdi[j, :, :] + dMdi[i, :, :] * M[j]).reshape(n, D)
            dSdi[j, i, :, :] = dSdi[i, j, :, :]
            dSdX[j, i, :, :] = dSdX[i, j, :, :]
        # end j

        S[i, i] = S[i, i] + 1e-06        # add small diagonal jitter for numerical reasons
    # end i

    # dSdX D+1 = -dSdX D+2  (MATLAB repeats this line twice)
    dSdX[:, :, D, :] = -dSdX[:, :, D + 1, :]
    dSdX[:, :, D, :] = -dSdX[:, :, D + 1, :]

    # 4) centralize moments
    S = S - M @ M.T

    # 5) Vectorize derivatives (Fortran order to match MATLAB column-major)
    P = D + 2
    dMds = dMds.reshape(E, D * D, order='F')
    dSdm = dSdm.reshape(E * E, D, order='F')
    dSds = dSds.reshape(E * E, D * D, order='F')
    # dVdm is from transpose; make C-contiguous before Fortran reshape
    dVdm = np.ascontiguousarray(dVdm).reshape(D * E, D, order='F')
    dVds = dVds.reshape(D * E, D * D, order='F')
    dMdi = dMdi.reshape(E, n * D, order='F')
    dMdt = dMdt.reshape(E, n * E, order='F')
    dMdX = dMdX.reshape(E, P * E, order='F')
    dSdi = dSdi.reshape(E * E, n * D, order='F')
    dSdt = dSdt.reshape(E * E, n * E, order='F')
    dSdX = dSdX.reshape(E * E, P * E, order='F')
    dVdi = dVdi.reshape(D * E, n * D, order='F')
    dVdt = dVdt.reshape(D * E, n * E, order='F')
    dVdX = dVdX.reshape(D * E, P * E, order='F')

    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdi, dSdi, dVdi, dMdt, dSdt, dVdt, dMdX, dSdX, dVdX
