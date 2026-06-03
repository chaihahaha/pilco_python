# loss_unicycle.py
# Robotic unicycle loss function. The loss is $1-\exp(-0.5*a*d^2)$, where
# $a$ is a (positive) constant and $d^2$ is the squared difference between
# the current configuration of the unicycle and a target set point.
#
# The mean and the variance of the loss are computed by averaging over the
# Gaussian distribution of the state $p(x) = \mathcal N(m,s)$ with mean $m$
# and covariance matrix $s$, plus cost.expl times the standard deviation of
# the loss (averaged wrt the same Gaussian), where the exploration paramater
# cost.expl defaults to zero.
#
#   def loss_unicycle(cost, m, s, compute_derivatives=True)
#
# *Input arguments:*
#
#   cost            cost structure (dict)
#     .p            parameters: [radius of wheel, length of rod]    [2]
#     .width        array of widths of the cost (summed together)
#     .expl         (optional) exploration parameter; default: 0
#   m               mean of state distribution                      [D]
#   s               covariance matrix for the state distribution    [D x D]
#
# *Output arguments:*
#
#   L     expected cost                                             [scalar]
#   dLdm  derivative of expected cost wrt. state mean vector        [D]
#   dLds  derivative of expected cost wrt. state covariance matrix  [D*D]
#   S2    variance of cost                                          [scalar]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-26
#
# High-Level Steps
# # Precomputations
# # Define static penalty as distance from target setpoint
# # Trigonometric augmentation
# # Calculate loss

import numpy as np
try:
    from ...util.gTrig import gTrig
except ImportError:
    from pilco_python.util.gTrig import gTrig
try:
    from ...loss.lossSat import lossSat
except ImportError:
    from pilco_python.loss.lossSat import lossSat


def _fillIn(S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, i_ind, k_ind, D):
    ## Code - replicated helper from loss_unicycle.m fillIn

    i_ind = np.asarray(i_ind).ravel().astype(int)
    k_ind = np.asarray(k_ind).ravel().astype(int)

    # Vectorized indices (column-major / Fortran order to match MATLAB)
    X = np.arange(D * D).reshape(D, D, order='F')
    XT = X.T

    # ii: linear indices of (i,i) pairs in column-major order
    # Use X[i_ind[:,None], i_ind] and ravel('F') to match MATLAB column-major extraction
    ii = X[np.ix_(i_ind, i_ind)].ravel(order='F')

    # kk: linear indices of (k,k) pairs
    kk = X[np.ix_(k_ind, k_ind)].ravel(order='F')

    # ik: linear indices of (i,k) pairs
    ik = X[np.ix_(i_ind, k_ind)].ravel(order='F')

    # ki: linear indices of (k,i) pairs
    ki = XT[np.ix_(i_ind, k_ind)].ravel(order='F')  # XT(i,k) = X(k,i)

    # chainrule
    Mdm[k_ind, :] = mdm @ Mdm[i_ind, :] + mds @ Sdm[ii, :]
    Mds[k_ind, :] = mdm @ Mds[i_ind, :] + mds @ Sds[ii, :]
    Sdm[kk, :] = sdm @ Mdm[i_ind, :] + sds @ Sdm[ii, :]
    Sds[kk, :] = sdm @ Mds[i_ind, :] + sds @ Sds[ii, :]
    dCdm = Cdm @ Mdm[i_ind, :] + Cds @ Sdm[ii, :]
    dCds = Cdm @ Mds[i_ind, :] + Cds @ Sds[ii, :]

    # off-diagonal blocks
    S[np.ix_(i_ind, k_ind)] = S[np.ix_(i_ind, i_ind)] @ C
    S[np.ix_(k_ind, i_ind)] = S[np.ix_(i_ind, k_ind)].T

    SS = np.kron(np.eye(len(k_ind)), S[np.ix_(i_ind, i_ind)])
    CC = np.kron(C.T, np.eye(len(i_ind)))

    Sdm[ik, :] = SS @ dCdm + CC @ Sdm[ii, :]
    Sdm[ki, :] = Sdm[ik, :]
    Sds[ik, :] = SS @ dCds + CC @ Sds[ii, :]
    Sds[ki, :] = Sds[ik, :]

    return S, Mdm, Mds, Sdm, Sds


def loss_unicycle(cost, m, s, compute_derivatives=True):
    ## Code

    m = np.asarray(m).ravel()
    s = np.atleast_2d(np.asarray(s))

    rw = cost['p'][0]
    r_val = cost['p'][1]

    if 'width' in cost:
        cw = np.asarray(cost['width']).ravel()
    else:
        cw = np.array([1.0])

    if 'expl' not in cost or cost['expl'] is None:
        b_val = 0.0
    else:
        b_val = float(cost['expl'])

    # Coordinates (0-based): theta and psif in the 10-dim dyno space
    # MATLAB: I6 = 8 (theta in dyno), I9 = 10 (psif in dyno)
    # MATLAB: Ixc = 6 (xc in dyno), Iyc = 7 (yc in dyno)
    # In Python 0-based dyno = [4,5,6,7,8,11,12,13,14,16]
    I6 = 7   # 0-based index of theta in dyno (MATLAB index 8 -> 7)
    I9 = 9   # 0-based index of psif in dyno (MATLAB index 10 -> 9)
    Ixc = 5  # 0-based index of xc in dyno (MATLAB index 6 -> 5)
    Iyc = 6  # 0-based index of yc in dyno (MATLAB index 7 -> 6)

    # 1. Some precomputations
    D = s.shape[1]                             # state dimension (dyno dim = 10)
    D0 = D + 2                                 # state dimension (augmented with I6-I9 and I6+I9)
    D1 = D0 + 8                                # state dimension (with sin/cos)
    L = 0.0
    S2 = 0.0
    dLdm = np.zeros(D)
    dLds = np.zeros(D * D)

    # MATLAB: P = [eye(D); zeros(2,D)]; P(D+1:end,I6) = [1;-1]; P(D+1:end,I9) = [1;1];
    P = np.zeros((D0, D))
    P[:D, :] = np.eye(D)
    P[D, I6] = 1.0
    P[D + 1, I6] = -1.0
    P[D, I9] = 1.0
    P[D + 1, I9] = 1.0

    # MATLAB: M = zeros(D1,1); M(1:D0) = P*m; S = zeros(D1); S(1:D0,1:D0) = P*s*P';
    M = np.zeros(D1)
    M[:D0] = P @ m.reshape(-1)
    S = np.zeros((D1, D1))
    S[:D0, :D0] = P @ s @ P.T

    # MATLAB: Mdm = [P; zeros(D1-D0,D)]; Sdm = zeros(D1*D1,D);
    Mdm = np.zeros((D1, D))
    Mdm[:D0, :] = P
    Sdm = np.zeros((D1 * D1, D))

    # MATLAB: Mds = zeros(D1,D*D); Sds = kron(Mdm,Mdm);
    Mds = np.zeros((D1, D * D))
    Sds = np.kron(Mdm, Mdm)

    # 2. Define static penalty as distance from target setpoint
    Q = np.zeros((D + 10, D + 10))
    C1 = np.array([rw, r_val / 2, r_val / 2])
    # MATLAB: Q([D+4 D+6 D+8],[D+4 D+6 D+8]) = 8*(C1'*C1);
    dz_idx = np.array([D + 3, D + 5, D + 7])  # 0-based: D+4 -> D+3, D+6 -> D+5, D+8 -> D+7
    Q[np.ix_(dz_idx, dz_idx)] = 8 * np.outer(C1, C1)

    C2 = np.array([1.0, -r_val])
    # MATLAB: Q([Ixc D+9],[Ixc D+9]) = 0.5*(C2'*C2);
    dx_idx = np.array([Ixc, D + 8])  # 0-based: D+9 -> D+8
    Q[np.ix_(dx_idx, dx_idx)] = 0.5 * np.outer(C2, C2)

    C3 = np.array([1.0, -(r_val + rw)])
    # MATLAB: Q([Iyc D+3],[Iyc D+3]) = 0.5*(C3'*C3);
    dy_idx = np.array([Iyc, D + 2])  # 0-based: D+3 -> D+2
    Q[np.ix_(dy_idx, dy_idx)] = 0.5 * np.outer(C3, C3)

    # MATLAB: Q(9,9) = (1/(4*pi))^2;  -- yaw angle loss
    Q[8, 8] = (1.0 / (4 * np.pi)) ** 2  # 0-based: index 9 -> 8

    target = np.zeros(D1)
    # MATLAB: target([D+4 D+6 D+8 D+10]) = 1;
    target[D + 3] = 1.0  # 0-based: D+4 -> D+3
    target[D + 5] = 1.0  # 0-based: D+6 -> D+5
    target[D + 7] = 1.0  # 0-based: D+8 -> D+7
    target[D + 9] = 1.0  # 0-based: D+10 -> D+9

    # 3. Trigonometric augmentation
    i_idx = np.arange(D0, dtype=int)   # 0:D0-1
    k_idx = np.arange(D0, D1, dtype=int)  # D0:D1-1

    # MATLAB: [M(k) S(k,k) C mdm sdm Cdm mds sds Cds] = gTrig(M(i),S(i,i),[I6 D+1 D+2 I9]);
    trig_indices = np.array([I6, D, D + 1, I9])
    M_k, V_kk, C_trig, mdm, sdm, Cdm, mds, sds, Cds = gTrig(
        M[i_idx], S[np.ix_(i_idx, i_idx)], trig_indices)

    M[k_idx] = M_k
    S[np.ix_(k_idx, k_idx)] = V_kk

    # MATLAB: [S Mdm Mds Sdm Sds] = fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,i,k,D1);
    S, Mdm, Mds, Sdm, Sds = _fillIn(
        S, C_trig, mdm, sdm, Cdm, mds, sds, Cds,
        Mdm, Sdm, Mds, Sds, i_idx, k_idx, D1)

    # 4. Calculate loss
    for cw_val in cw:                  # scale mixture of immediate costs
        cost_zW = {'z': target, 'W': Q / (cw_val ** 2)}
        r, rdM, rdS, s2, s2dM, s2dS, _, _, _ = lossSat(cost_zW, M, S, compute_derivatives=True)

        L = L + r
        S2 = S2 + s2
        dLdm = dLdm + rdM.ravel() @ Mdm + rdS.ravel() @ Sdm
        dLds = dLds + rdM.ravel() @ Mds + rdS.ravel() @ Sds

        if b_val != 0.0 and abs(s2) > 1e-12:
            L = L + b_val * np.sqrt(s2)
            dLdm = dLdm + b_val / np.sqrt(s2) * (s2dM.ravel() @ Mdm + s2dS.ravel() @ Sdm) / 2.0
            dLds = dLds + b_val / np.sqrt(s2) * (s2dM.ravel() @ Mds + s2dS.ravel() @ Sds) / 2.0

    # normalize
    n = len(cw)
    L = L / n
    dLdm = dLdm / n
    dLds = dLds / n
    S2 = S2 / n

    if not compute_derivatives:
        return L, np.array([], dtype=float), np.array([], dtype=float), S2

    return L, dLdm, dLds, S2
