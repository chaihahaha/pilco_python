# loss_pendulum.py
# *Summary:* Pendulum loss function; the loss is
# $1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
# between the actual and desired position of the tip of the pendulum.
# The mean and the variance of the loss are computed by averaging over the
# Gaussian distribution of the state $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$.
# Derivatives of these quantities are computed when desired.
#
#
#    def loss_pendulum(cost, m, s)
#
#
# *Input arguments:*
#
#   cost            cost dict
#     .p            length of the pendulum                           [1 x  1 ]
#     .width        array of widths of the cost (summed together)
#     .expl         (optional) exploration parameter
#     .angle        (optional) array of angle indices (0-indexed)
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
# Last modified: 2014-01-05
#
## High-Level Steps
# # Precomputations
# # Define static penalty as distance from target setpoint
# # Trigonometric augmentation
# # Calculate loss

import numpy as np
from ...util.gTrig import gTrig
from ...loss.lossSat import lossSat


def loss_pendulum(cost, m, s):
    ## Code

    # Defaults
    if 'width' in cost and cost['width'] is not None:
        cw = np.atleast_1d(cost['width']).ravel()
    else:
        cw = np.array([1.0])

    if 'expl' not in cost or cost['expl'] is None or np.isscalar(cost['expl']) and cost['expl'] == 0:
        b = 0.0
    else:
        b = float(cost['expl'])

    m = np.atleast_2d(np.asarray(m).ravel()).T.ravel()
    s = np.atleast_2d(np.asarray(s))

    # 1. Some precomputations
    D0 = s.shape[1]                               # state dimension
    D1 = D0 + 2 * len(cost['angle'])              # state dimension (with sin/cos)

    M = np.zeros(D1)
    M[:D0] = m
    S = np.zeros((D1, D1))
    S[:D0, :D0] = s
    Mdm = np.vstack([np.eye(D0), np.zeros((D1 - D0, D0))])
    Sdm = np.zeros((D1 * D1, D0))
    Mds = np.zeros((D1, D0 * D0))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    ell = cost['p']
    Q = np.zeros((D1, D1))
    Q[D0:D0 + 2, D0:D0 + 2] = np.eye(2) * ell**2

    # 3. Trigonometric augmentation
    i = np.arange(D0)                              # 0-indexed, same as MATLAB 1:D0→0:D0-1
    k = np.arange(D0, D1)                          # D0+1:D1 → D0:D1-1

    if D1 - D0 > 0:
        target_aug = np.concatenate([
            cost['target'].ravel(),
            gTrig(cost['target'].ravel(), np.zeros((len(cost['target']), len(cost['target']))),
                  cost['angle'],
                  compute_derivatives=False)[0]
        ])

        M_k, S_kk, C, mdm, sdm, Cdm, mds, sds, Cds = \
            gTrig(M[i], S[np.ix_(i, i)], cost['angle'], compute_derivatives=True)

        M[k] = M_k
        S[np.ix_(k, k)] = S_kk

        # Fill in covariance matrix and derivatives
        S, Mdm, Sdm, Mds, Sds = \
            _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds,
                     Mdm, Sdm, Mds, Sds, i, k, D1)
    else:
        target_aug = cost['target'].ravel().copy()

    # 4. Calculate loss
    L = 0.0
    dLdm = np.zeros(D0)
    dLds = np.zeros(D0 * D0)
    S2 = 0.0

    nw = len(cw)
    for iw in range(nw):                         # scale mixture of immediate costs
        cost_copy = dict(cost)
        cost_copy['z'] = target_aug.copy()
        cost_copy['W'] = Q / cw[iw]**2

        r, rdM, rdS, s2, s2dM, s2dS, _, _, _ = lossSat(cost_copy, M, S, compute_derivatives=True)

        L = L + r
        S2 = S2 + s2
        dLdm = dLdm + rdM.ravel() @ Mdm + rdS.ravel('F') @ Sdm
        dLds = dLds + rdM.ravel() @ Mds + rdS.ravel('F') @ Sds

        if b != 0.0 and abs(s2) > 1e-12:
            L = L + b * np.sqrt(s2)
            dLdm = dLdm + b / np.sqrt(s2) * (s2dM.ravel() @ Mdm + s2dS.ravel('F') @ Sdm) / 2
            dLds = dLds + b / np.sqrt(s2) * (s2dM.ravel() @ Mds + s2dS.ravel('F') @ Sds) / 2

    # normalize
    L = float(L) / nw
    dLdm = dLdm / nw
    dLds = dLds / nw
    S2 = float(S2) / nw

    return L, dLdm, dLds, S2


def _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds,
             Mdm, Sdm, Mds, Sds, i, k, D):
    """Fill in covariance matrix and derivatives after gTrig augmentation.

    This is the translation of the nested fillIn function from loss_pendulum.m.
    All indices are 0-indexed (converted from MATLAB 1-indexed).
    """
    # Vectorized indices (column-major / Fortran order, matching MATLAB)
    X = np.arange(1, D * D + 1).reshape(D, D, order='F')   # 1-indexed like MATLAB
    XT = X.T

    X_flat = X.ravel(order='F')
    XT_flat = XT.ravel(order='F')

    I_temp = np.zeros((D, D))
    I_temp[np.ix_(i, i)] = 1
    ii = X_flat[I_temp.ravel(order='F') == 1] - 1           # 0-indexed

    I_temp = np.zeros((D, D))
    I_temp[np.ix_(k, k)] = 1
    kk = X_flat[I_temp.ravel(order='F') == 1] - 1

    I_temp = np.zeros((D, D))
    I_temp[np.ix_(i, k)] = 1
    ik = X_flat[I_temp.ravel(order='F') == 1] - 1

    I_temp = np.zeros((D, D))
    I_temp[np.ix_(k, i)] = 1
    ki = XT_flat[I_temp.ravel(order='F') == 1] - 1

    # chainrule
    Mdm[k, :] = mdm @ Mdm[i, :] + mds @ Sdm[ii, :]
    Mds[k, :] = mdm @ Mds[i, :] + mds @ Sds[ii, :]
    Sdm[kk, :] = sdm @ Mdm[i, :] + sds @ Sdm[ii, :]
    Sds[kk, :] = sdm @ Mds[i, :] + sds @ Sds[ii, :]
    dCdm = Cdm @ Mdm[i, :] + Cds @ Sdm[ii, :]
    dCds = Cdm @ Mds[i, :] + Cds @ Sds[ii, :]

    # off-diagonal covariance
    S[np.ix_(i, k)] = S[np.ix_(i, i)] @ C
    S[np.ix_(k, i)] = S[np.ix_(i, k)].T

    SS = np.kron(np.eye(len(k)), S[np.ix_(i, i)])
    CC = np.kron(C.T, np.eye(len(i)))
    Sdm[ik, :] = SS @ dCdm + CC @ Sdm[ii, :]
    Sdm[ki, :] = Sdm[ik, :]
    Sds[ik, :] = SS @ dCds + CC @ Sds[ii, :]
    Sds[ki, :] = Sds[ik, :]

    return S, Mdm, Sdm, Mds, Sds
