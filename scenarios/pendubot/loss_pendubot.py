# loss_pendubot.py
# *Summary:* Pendubot loss function; the loss is
# $1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
# between the actual and desired position of the tip of the outer pendulum.
# The mean and the variance of the loss are computed by averaging over the
# Gaussian distribution of the state $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$.
# Derivatives of these quantities are computed when desired.
#
#
#   def loss_pendubot(cost, m, s, compute_derivatives=True)
#
#
# *Input arguments:*
#
#   cost            cost structure (dict)
#     .p            lengths of the 2 pendulums                      [2 x  1 ]
#     .width        array of widths of the cost (summed together)
#     .expl         (optional) exploration parameter
#     .angle        (optional) array of angle indices
#     .target       target state                                    [4 x  1 ]
#   m               mean of state distribution                      [4 x  1 ]
#   s               covariance matrix for the state distribution    [4 x  4 ]
#
# *Output arguments:*
#
#   L     expected cost                                             [1 x  1 ]
#   dLdm  derivative of expected cost wrt. state mean vector        [1 x  4 ]
#   dLds  derivative of expected cost wrt. state covariance matrix  [1 x 16 ]
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
from ...util.gTrig import gTrig
from ...loss.lossSat import lossSat


def loss_pendubot(cost, m, s, compute_derivatives=True):
    ## Code
    m = np.asarray(m).ravel()
    s = np.atleast_2d(s)

    cw = cost.get('width', 1.0)
    if isinstance(cw, (int, float)):
        cw = np.array([cw])
    else:
        cw = np.asarray(cw).ravel()

    if 'expl' not in cost or cost['expl'] is None or cost['expl'] == 0:
        b = 0.0
    else:
        b = cost['expl']

    # 1. Some precomputations
    D0 = s.shape[1]
    D = D0
    D1 = D0 + 2 * len(cost['angle'])
    Dk = D1 - D0  # = 2 * len(cost['angle'])

    M = np.zeros(D1)
    M[:D0] = m
    S = np.zeros((D1, D1))
    S[:D0, :D0] = s

    Mdm = np.vstack([np.eye(D0), np.zeros((Dk, D0))])
    Sdm = np.zeros((D1 * D1, D0))
    Mds = np.zeros((D1, D0 * D0))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    ell1 = cost['p'][0]
    ell2 = cost['p'][1]
    C_mat = np.array([[ell1, 0, ell2, 0],
                       [0, ell1, 0, ell2]])
    Q = np.zeros((D1, D1))
    Q[D:D+4, D:D+4] = C_mat.T @ C_mat

    # 3. Trigonometric augmentation
    target = np.asarray(cost['target']).ravel()
    i_idx = np.arange(D0)
    k_idx = np.arange(D0, D1)

    target_aug, _, _ = gTrig(target, np.zeros((D0, D0)), cost['angle'],
                              compute_derivatives=False)
    target_full = np.concatenate([target, target_aug])

    if compute_derivatives:
        M_kk, S_kk, C_trig, mdm, sdm, Cdm, mds, sds, Cds = \
            gTrig(M[i_idx], S[np.ix_(i_idx, i_idx)], cost['angle'],
                  compute_derivatives=True)
    else:
        M_kk, S_kk, C_trig = \
            gTrig(M[i_idx], S[np.ix_(i_idx, i_idx)], cost['angle'],
                  compute_derivatives=False)

    M[k_idx] = M_kk
    S[np.ix_(k_idx, k_idx)] = S_kk

    # cross-covariance off-diagonal blocks
    S_ik = S[np.ix_(i_idx, i_idx)] @ C_trig
    S[np.ix_(i_idx, k_idx)] = S_ik
    S[np.ix_(k_idx, i_idx)] = S_ik.T

    if compute_derivatives:
        # Compute vectorized linear indices (MATLAB: reshape(1:D*D,[D D]))
        X = np.arange(D1 * D1).reshape(D1, D1, order='F')

        # ii: indices for i,i block (D0 x D0)
        ii = X[np.ix_(i_idx, i_idx)].ravel(order='F')
        # kk: indices for k,k block (Dk x Dk)
        kk = X[np.ix_(k_idx, k_idx)].ravel(order='F')
        # ik: indices for i,k block (D0 x Dk)
        ik = X[np.ix_(i_idx, k_idx)].ravel(order='F')
        # ki: indices for k,i block (Dk x D0) -- note: XT in MATLAB
        ki = X[np.ix_(k_idx, i_idx)].ravel(order='F')

        # chainrule for derivatives
        Mdm[np.ix_(k_idx)] = mdm @ Mdm[np.ix_(i_idx)] + mds @ Sdm[ii, :]
        Mds[np.ix_(k_idx)] = mdm @ Mds[np.ix_(i_idx)] + mds @ Sds[ii, :]
        Sdm[kk, :] = sdm @ Mdm[np.ix_(i_idx)] + sds @ Sdm[ii, :]
        Sds[kk, :] = sdm @ Mds[np.ix_(i_idx)] + sds @ Sds[ii, :]
        dCdm = Cdm @ Mdm[np.ix_(i_idx)] + Cds @ Sdm[ii, :]
        dCds = Cdm @ Mds[np.ix_(i_idx)] + Cds @ Sds[ii, :]

        # off-diagonal covariance derivative blocks
        # SS = kron(eye(length(k)), S(i,i))
        SS_kron = np.kron(np.eye(Dk), S[np.ix_(i_idx, i_idx)])
        # CC = kron(C', eye(length(i)))
        CC_kron = np.kron(C_trig.T, np.eye(D0))

        Sdm[ik, :] = SS_kron @ dCdm + CC_kron @ Sdm[ii, :]
        Sdm[ki, :] = Sdm[ik, :]
        Sds[ik, :] = SS_kron @ dCds + CC_kron @ Sds[ii, :]
        Sds[ki, :] = Sds[ik, :]

    # 4. Calculate loss
    L = 0.0
    dLdm = np.zeros((1, D0))
    dLds = np.zeros((1, D0 * D0))
    S2 = 0.0

    for ci_idx in range(len(cw)):
        cost_i = {'z': target_full, 'W': Q / cw[ci_idx]**2}
        r, rdM, rdS, s2_val, s2dM, s2dS, _, _, _ = lossSat(cost_i, M, S)

        L = L + r
        S2 = S2 + s2_val
        if compute_derivatives:
            dLdm = dLdm + rdM.ravel() @ Mdm + rdS.ravel() @ Sdm
            dLds = dLds + rdM.ravel() @ Mds + rdS.ravel() @ Sds

            if b != 0.0 and abs(s2_val) > 1e-12:
                L = L + b * np.sqrt(s2_val)
                dLdm = dLdm + b / np.sqrt(s2_val) * \
                    (s2dM.ravel() @ Mdm + s2dS.ravel() @ Sdm) / 2.0
                dLds = dLds + b / np.sqrt(s2_val) * \
                    (s2dM.ravel() @ Mds + s2dS.ravel() @ Sds) / 2.0

    # normalize
    n = len(cw)
    L = L / n
    dLdm = dLdm / n
    dLds = dLds / n
    S2 = S2 / n

    if not compute_derivatives:
        return L, dLdm.ravel(), dLds.ravel(), S2

    return L, dLdm.ravel(), dLds.ravel(), S2
