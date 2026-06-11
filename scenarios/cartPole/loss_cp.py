# loss_cp.py
# *Summary:* Cart-Pole loss function; the loss is
# $1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
# between the actual and desired position of tip of the pendulum.
# The mean and the variance of the loss are computed by averaging over the
# Gaussian state distribution $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$.
# Derivatives of these quantities are computed when desired.
#
#   def loss_cp(cost, m, s)
#
# *Input arguments:*
#
#   cost            cost structure (dict)
#     .p            length of pendulum                              [1 x  1 ]
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
# Last modified: 2013-05-16
#
## High-Level Steps
# # Precomputations
# # Define static penalty as distance from target setpoint
# # Trigonometric augmentation
# # Calculate loss

import numpy as np
from pilco_python.util.gTrig import gTrig
from pilco_python.loss.lossSat import lossSat


def loss_cp(cost, m, s):
    # Code

    if isinstance(cost.get('width'), (list, np.ndarray)):
        cw = np.atleast_1d(np.asarray(cost['width'], dtype=float)).ravel()
    else:
        cw = np.array([1.0]) if 'width' not in cost else np.atleast_1d(np.asarray(cost['width'], dtype=float)).ravel()

    if 'expl' not in cost or cost['expl'] is None or cost['expl'] == 0:
        b = 0
    else:
        b = cost['expl']

    # 1. Some precomputations
    D0 = s.shape[1]                           # state dimension
    D1 = D0 + 2 * len(cost['angle'])           # state dimension (with sin/cos)

    M = np.zeros(D1); M[:D0] = m.ravel()
    S = np.zeros((D1, D1)); S[:D0, :D0] = s
    Mdm = np.vstack([np.eye(D0), np.zeros((D1 - D0, D0))])
    Sdm = np.zeros((D1 * D1, D0))
    Mds = np.zeros((D1, D0 * D0))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    ell = cost['p']  # pendulum length
    Q = np.zeros((D1, D1))
    Q[np.ix_([0, D0], [0, D0])] = np.outer([1, ell], [1, ell])  # [1 ell]'*[1 ell]
    Q[D0 + 1, D0 + 1] = ell**2

    # 3. Trigonometric augmentation
    if D1 - D0 > 0:
        # augment target
        target_raw = np.asarray(cost['target']).ravel()
        target_gTrig = gTrig(target_raw, 0 * np.atleast_2d(s), cost['angle'], compute_derivatives=False)[0]
        target = np.concatenate([target_raw, target_gTrig])

        # augment state
        i = np.arange(D0); k = np.arange(D0, D1)

        Mkk, Skk, C, mdm, sdm, Cdm, mds, sds, Cds = gTrig(M[i], S[np.ix_(i, i)], cost['angle'], compute_derivatives=True)

        M[k] = Mkk
        S[np.ix_(k, k)] = Skk

        # compute derivatives (for augmentation)
        X = np.arange(1, D1 * D1 + 1).reshape((D1, D1), order='F')
        XT = X.T.copy()
        I = np.zeros((D1, D1)); I[np.ix_(i, i)] = 1
        I_flat = I.ravel(order='F')
        ii = X.ravel(order='F')[I_flat == 1].astype(int)
        I = np.zeros((D1, D1)); I[np.ix_(k, k)] = 1
        I_flat = I.ravel(order='F')
        kk = X.ravel(order='F')[I_flat == 1].astype(int)
        I = np.zeros((D1, D1)); I[np.ix_(i, k)] = 1
        I_flat = I.ravel(order='F')
        ik = X.ravel(order='F')[I_flat == 1].astype(int)
        ki = XT.ravel(order='F')[I_flat == 1].astype(int)

        # 0-based indexing for Python
        ii = ii - 1; kk = kk - 1; ik = ik - 1; ki = ki - 1

        Mdm[k, :] = mdm @ Mdm[i, :] + mds @ Sdm[ii, :]
        Mds[k, :] = mdm @ Mds[i, :] + mds @ Sds[ii, :]
        Sdm[kk, :] = sdm @ Mdm[i, :] + sds @ Sdm[ii, :]
        Sds[kk, :] = sdm @ Mds[i, :] + sds @ Sds[ii, :]
        dCdm = Cdm @ Mdm[i, :] + Cds @ Sdm[ii, :]
        dCds = Cdm @ Mds[i, :] + Cds @ Sds[ii, :]

        # off-diagonal
        S[np.ix_(i, k)] = S[np.ix_(i, i)] @ C
        S[np.ix_(k, i)] = S[np.ix_(i, k)].T

        SS = np.kron(np.eye(len(k)), S[np.ix_(i, i)])
        CC_mat = np.kron(C.T, np.eye(len(i)))
        Sdm[ik, :] = SS @ dCdm + CC_mat @ Sdm[ii, :]
        Sdm[ki, :] = Sdm[ik, :]
        Sds[ik, :] = SS @ dCds + CC_mat @ Sds[ii, :]
        Sds[ki, :] = Sds[ik, :]
    else:
        target = np.asarray(cost['target']).ravel()

    # 4. Calculate loss!
    L = 0.0
    dLdm = np.zeros((1, D0))
    dLds = np.zeros((1, D0 * D0))
    S2 = 0.0
    dS2dm = np.zeros((1, D0))
    dS2ds = np.zeros((1, D0 * D0))
    for i_idx in range(len(cw)):                    # scale mixture of immediate costs
        cost_sat = {'z': target, 'W': Q / cw[i_idx]**2}
        r, rdM, rdS, s2, s2dM, s2dS, _, _, _ = lossSat(cost_sat, M, S)

        L = L + r
        S2 = S2 + s2
        dLdm = dLdm + rdM.ravel()[np.newaxis, :] @ Mdm + rdS.ravel()[np.newaxis, :] @ Sdm
        dLds = dLds + rdM.ravel()[np.newaxis, :] @ Mds + rdS.ravel()[np.newaxis, :] @ Sds
        dS2dm = dS2dm + s2dM.ravel()[np.newaxis, :] @ Mdm + s2dS.ravel()[np.newaxis, :] @ Sdm
        dS2ds = dS2ds + s2dM.ravel()[np.newaxis, :] @ Mds + s2dS.ravel()[np.newaxis, :] @ Sds

        if b != 0 and abs(s2) > 1e-12:
            L = L + b * np.sqrt(s2)
            dLdm = dLdm + b / np.sqrt(s2) * (s2dM.ravel()[np.newaxis, :] @ Mdm + s2dS.ravel()[np.newaxis, :] @ Sdm) / 2
            dLds = dLds + b / np.sqrt(s2) * (s2dM.ravel()[np.newaxis, :] @ Mds + s2dS.ravel()[np.newaxis, :] @ Sds) / 2

    # normalize
    n = len(cw)
    L = L / n
    dLdm = dLdm / n
    dLds = dLds / n
    S2 = S2 / n
    dS2dm = dS2dm / n
    dS2ds = dS2ds / n

    return L, dLdm.ravel(), dLds.ravel(), S2, dS2dm.ravel(), dS2ds.ravel()
