# loss_cdp.py
# *Summary:* Cart-Double-Pendulum loss function; the loss is
# $1-\exp(-0.5*d^2*a)$,  where $a>0$ and  $d^2$ is the squared difference
# between the actual and desired position of the end of the outer pendulum.
# The mean and the variance of the loss are computed by averaging over the
# Gaussian distribution of the state $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$.
# Derivatives of these quantities are computed when desired.
#
#
#   def loss_cdp(cost, m, s, compute_derivatives=True)
#
#
# *Input arguments:*
#
#   cost            cost structure
#     'p'           lengths of the 2 pendulums                      [2 x  1 ]
#     'width'       array of widths of the cost (summed together)
#     'expl'        (optional) exploration parameter
#     'angle'       (optional) array of angle indices
#     'target'      target state                                    [D x  1 ]
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
# Last modified: 2013-03-07
#
## High-Level Steps
# # Precomputations
# # Define static penalty as distance from target setpoint
# # Trigonometric augmentation
# # Calculate loss
#
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
try:
    from ...util.gTrig import gTrig
    from ...loss.lossSat import lossSat
except (ImportError, ValueError):
    from pilco_python.util.gTrig import gTrig
    from pilco_python.loss.lossSat import lossSat


def loss_cdp(cost, m, s, compute_derivatives=True):
    ## Code

    cw = cost.get('width', 1)
    b = cost.get('expl', 0)
    if b is None:
        b = 0

    # 1. Some precomputations
    D0 = s.shape[1]
    D = D0                                             # state dimension
    angle_idx = np.asarray(cost['angle']).ravel().astype(int)
    D1 = D0 + 2*len(angle_idx)        # state dimension (with sin/cos)

    M = np.zeros(D1); M[:D0] = m.ravel()
    S = np.zeros((D1, D1)); S[:D0, :D0] = s
    Mdm = np.vstack([np.eye(D0), np.zeros((D1-D0, D0))])
    Sdm = np.zeros((D1*D1, D0))
    Mds = np.zeros((D1, D0*D0))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    target_base = np.asarray(cost['target']).ravel()
    target_trig = gTrig(target_base, 0*s, angle_idx, compute_derivatives=False)[0]
    target = np.concatenate([target_base, target_trig])

    ell1 = cost['p'][0]
    ell2 = cost['p'][1]
    C_mat = np.array([[1, -ell1, 0, -ell2, 0],
                      [0, 0, ell1, 0, ell2]])

    Q = np.zeros((D1, D1))
    idx_q = [0, D, D+1, D+2, D+3]      # [1, D+1:D+4] in MATLAB 1-indexed
    for pi, ri in enumerate(idx_q):
        for pj, rj in enumerate(idx_q):
            Q[ri, rj] = (C_mat.T @ C_mat)[pi, pj]

    # 3. Trigonometric augmentation
    i = np.arange(D0); k = np.arange(D0, D1)

    Mkk, Skk, C, mdm, sdm, Cdm, mds, sds, Cds = \
        gTrig(M[i], S[np.ix_(i, i)], angle_idx, compute_derivatives=True)
    M[k] = Mkk
    S[np.ix_(k, k)] = Skk
    S, Mdm, Mds, Sdm, Sds = \
        _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
                 i, i, k, D1)

    # 4. Calculate loss
    L = 0; dLdm = np.zeros((1, D0)); dLds = np.zeros((1, D0*D0)); S2 = 0
    dS2dm = np.zeros((1, D0)); dS2ds = np.zeros((1, D0*D0))

    cw_arr = np.atleast_1d(cw)
    for ci in range(len(cw_arr)):
        cost_mod = dict(cost)
        cost_mod['z'] = target
        cost_mod['W'] = Q / cw_arr[ci]**2

        r, rdM, rdS, s2, s2dM, s2dS, _, _, _ = lossSat(cost_mod, M, S, compute_derivatives=True)

        L = L + r
        S2 = S2 + s2
        dLdm = dLdm + rdM.ravel() @ Mdm + rdS.ravel() @ Sdm
        dLds = dLds + rdM.ravel() @ Mds + rdS.ravel() @ Sds
        dS2dm = dS2dm + s2dM.ravel() @ Mdm + s2dS.ravel() @ Sdm
        dS2ds = dS2ds + s2dM.ravel() @ Mds + s2dS.ravel() @ Sds

        b_val = float(b)
        if abs(b_val) > 0 and abs(s2) > 1e-12:
            L = L + b_val*np.sqrt(s2)
            dLdm = dLdm + b_val/np.sqrt(s2) * (s2dM.ravel() @ Mdm + s2dS.ravel() @ Sdm) / 2
            dLds = dLds + b_val/np.sqrt(s2) * (s2dM.ravel() @ Mds + s2dS.ravel() @ Sds) / 2

    n = len(cw_arr)
    L = L / n; dLdm = dLdm / n; dLds = dLds / n; S2 = S2 / n
    dS2dm = dS2dm / n; dS2ds = dS2ds / n

    if not compute_derivatives:
        return L, dLdm, dLds, S2, dS2dm, dS2ds

    return L, dLdm, dLds, S2, dS2dm, dS2ds


# Fill in covariance matrix...and derivatives ----------------------------
def _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
             i, j, k, D):
    X = np.arange(1, D*D+1).reshape((D, D), order='F')
    XT = X.T.copy()

    I_mat = np.zeros((D, D)); I_mat[np.ix_(i, i)] = 1
    ii_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]

    I_mat = np.zeros((D, D)); I_mat[np.ix_(k, k)] = 1
    kk_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]

    I_mat = np.zeros((D, D)); I_mat[np.ix_(i, k)] = 1
    ik_vec = X.ravel(order='F')[I_mat.ravel(order='F') == 1]

    I_mat = np.zeros((D, D)); I_mat[np.ix_(k, i)] = 1
    ki_vec = XT.ravel(order='F')[I_mat.ravel(order='F') == 1]

    ii_idx = ii_vec.astype(int) - 1
    kk_idx = kk_vec.astype(int) - 1
    ik_idx = ik_vec.astype(int) - 1
    ki_idx = ki_vec.astype(int) - 1

    Mdm[k, :] = mdm @ Mdm[i, :] + mds @ Sdm[ii_idx, :]                      # chainrule
    Mds[k, :] = mdm @ Mds[i, :] + mds @ Sds[ii_idx, :]
    Sdm[kk_idx, :] = sdm @ Mdm[i, :] + sds @ Sdm[ii_idx, :]
    Sds[kk_idx, :] = sdm @ Mds[i, :] + sds @ Sds[ii_idx, :]
    dCdm = Cdm @ Mdm[i, :] + Cds @ Sdm[ii_idx, :]
    dCds = Cdm @ Mds[i, :] + Cds @ Sds[ii_idx, :]

    S[np.ix_(i, k)] = S[np.ix_(i, i)] @ C; S[np.ix_(k, i)] = S[np.ix_(i, k)].T  # off-diagonal
    SS = np.kron(np.eye(len(k)), S[np.ix_(i, i)])
    CC = np.kron(C.T, np.eye(len(i)))
    Sdm[ik_idx, :] = SS @ dCdm + CC @ Sdm[ii_idx, :]; Sdm[ki_idx, :] = Sdm[ik_idx, :]
    Sds[ik_idx, :] = SS @ dCds + CC @ Sds[ii_idx, :]; Sds[ki_idx, :] = Sds[ik_idx, :]

    return S, Mdm, Mds, Sdm, Sds
