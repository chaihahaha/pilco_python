# loss_dp.py
# *Summary:* Double-Pendulum loss function; the loss is
# $1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
# between the actual and desired position of the tip of the outer pendulum.
# The mean and the variance of the loss are computed by averaging over the
# Gaussian distribution of the state $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$.
# Derivatives of these quantities are computed when desired.
#
#
#   def loss_dp(cost, m, s)
#
#
# *Input arguments:*
#
#   cost            cost structure
#     .p            lengths of the 2 pendulums                      [2 x  1 ]
#     .width        array of widths of the cost (summed together)
#     .expl         (optional) exploration parameter
#     .angle        (optional) array of angle indices
#     .target       target state                                    [D x  1 ]
#   m               mean of state distribution                      [D x  1 ]
#   s               covariance matrix for the state distribution    [D x  D ]
#
# *Output arguments:*
#
#   L     expected cost                                             [1 x  1 ]
#   dLdm  derivative of expected cost wrt. state mean vector        [1 x  D ]
#   dLds  derivative of expected cost wrt. state covariance matrix  [1 x D^2]
#   S2    variance of cost                                          [1 x  1 ]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-08
#
## High-Level Steps
# # Precomputations
# # Define static penalty as distance from target setpoint
# # Trigonometric augmentation
# # Calculate loss

import numpy as np
from pilco_python.util.gTrig import gTrig
from pilco_python.loss.lossSat import lossSat


def loss_dp(cost, m, s):
    ## Code

    if 'width' in cost and cost['width'] is not None:
        cw = cost['width']
    else:
        cw = 1
    cw = np.atleast_1d(np.asarray(cw, dtype=float)).ravel()

    if 'expl' not in cost or cost['expl'] is None:
        b = 0
    else:
        b = cost['expl']

    # 1. Some precomputations
    D0 = s.shape[1]
    D = D0                                  # state dimension
    D1 = D0 + 2 * len(cost['angle'])        # state dimension (with sin/cos)

    M = np.zeros(D1)
    M[:D0] = np.asarray(m).ravel()
    S = np.zeros((D1, D1))
    S[:D0, :D0] = np.atleast_2d(s)
    Mdm = np.vstack([np.eye(D0), np.zeros((D1 - D0, D0))])
    Sdm = np.zeros((D1 * D1, D0))
    Mds = np.zeros((D1, D0 * D0))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    ell1 = cost['p'][0]
    ell2 = cost['p'][1]
    C = np.array([[ell1, 0, ell2, 0], [0, ell1, 0, ell2]])
    Q = np.zeros((D1, D1))
    Q[D:D+4, D:D+4] = C.T @ C

    # pass Q via cost for now; lossSat uses cost.W = Q/cw^2 later
    # 3. Trigonometric augmentation
    if D1 - D0 > 0:
        target_cost = np.asarray(cost['target']).ravel()
        target = np.concatenate([target_cost,
                                 gTrig(target_cost, np.zeros_like(s), cost['angle'],
                                       compute_derivatives=False)[0]])

        i = np.arange(D0)
        k = np.arange(D0, D1)

        gtrig_out = gTrig(M[i], S[np.ix_(i, i)], cost['angle'], compute_derivatives=True)
        Mkk, Skk, Cmat, mdm, sdm, Cdm, mds, sds, Cds = gtrig_out
        M[k] = Mkk
        S[np.ix_(k, k)] = Skk
        S, Mdm, Mds, Sdm, Sds = _fill_in(
            S, Cmat, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, i, k, D1)

    # 4. Calculate loss
    L = 0.0
    dLdm = np.zeros(D0)
    dLds = np.zeros(D0 * D0)
    S2 = 0.0
    for i_idx in range(len(cw)):            # scale mixture of immediate costs
        cost_iter = dict(cost)
        cost_iter['z'] = target
        cost_iter['W'] = Q / (cw[i_idx]**2)
        r, rdM, rdS, s2, s2dM, s2dS = lossSat(cost_iter, M, S)[:6]

        L = L + r
        S2 = S2 + s2
        # rdM: (D1,) or (1, D1) -> (D1,). rdS: (D1, D1) -> flattened Fortran order
        dLdm = dLdm + rdM.ravel() @ Mdm + rdS.ravel(order='F') @ Sdm
        dLds = dLds + rdM.ravel() @ Mds + rdS.ravel(order='F') @ Sds

        if b != 0 and abs(s2) > 1e-12:
            L = L + b * np.sqrt(s2)
            dLdm = dLdm + b / np.sqrt(s2) * (s2dM.ravel() @ Mdm + s2dS.ravel(order='F') @ Sdm) / 2
            dLds = dLds + b / np.sqrt(s2) * (s2dM.ravel() @ Mds + s2dS.ravel(order='F') @ Sds) / 2

    # normalize
    n = len(cw)
    L = L / n
    dLdm = dLdm / n
    dLds = dLds / n
    S2 = S2 / n

    return L.item(), dLdm.reshape(1, -1), dLds.reshape(1, -1), S2.item()


def _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, i, k, D):
    """Fill in covariance matrix...and derivatives"""
    X = np.arange(1, D * D + 1).reshape((D, D), order='F')
    XT = X.T
    I_mat = np.zeros((D, D))
    I_mat[np.ix_(i, i)] = 1
    ii_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]
    I_mat = np.zeros((D, D))
    I_mat[np.ix_(k, k)] = 1
    kk_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]
    I_mat = np.zeros((D, D))
    I_mat[np.ix_(i, k)] = 1
    ik_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]
    I_mat = np.zeros((D, D))
    I_mat[np.ix_(k, i)] = 1
    ki_vec = XT.ravel(order='F')[I_mat.ravel(order='F') == 1]

    ii_idx = ii_vec.astype(int) - 1
    kk_idx = kk_vec.astype(int) - 1
    ik_idx = ik_vec.astype(int) - 1
    ki_idx = ki_vec.astype(int) - 1

    Mdm[np.ix_(k, np.arange(Mdm.shape[1]))] = \
        mdm @ Mdm[i, :] + mds @ Sdm[ii_idx, :]                      # chainrule
    Mds[np.ix_(k, np.arange(Mds.shape[1]))] = \
        mdm @ Mds[i, :] + mds @ Sds[ii_idx, :]
    Sdm[kk_idx, :] = sdm @ Mdm[i, :] + sds @ Sdm[ii_idx, :]
    Sds[kk_idx, :] = sdm @ Mds[i, :] + sds @ Sds[ii_idx, :]
    dCdm = Cdm @ Mdm[i, :] + Cds @ Sdm[ii_idx, :]
    dCds = Cdm @ Mds[i, :] + Cds @ Sds[ii_idx, :]

    S[np.ix_(i, k)] = S[np.ix_(i, i)] @ C
    S[np.ix_(k, i)] = S[np.ix_(i, k)].T                        # off-diagonal
    SS = np.kron(np.eye(len(k)), S[np.ix_(i, i)])
    CC = np.kron(C.T, np.eye(len(i)))
    Sdm[ik_idx, :] = SS @ dCdm + CC @ Sdm[ii_idx, :]
    Sdm[ki_idx, :] = Sdm[ik_idx, :]
    Sds[ik_idx, :] = SS @ dCds + CC @ Sds[ii_idx, :]
    Sds[ki_idx, :] = Sds[ik_idx, :]

    return S, Mdm, Mds, Sdm, Sds
