# conCat.py
# *Summary:* Compute a control signal $u$ from a state distribution
# $x\sim\mathcal N(x|m,s)$. Here, the predicted control distribution
# and its derivatives are computed by concatenating a controller "con" with
# a saturation function "sat", such as gSat.py.
#
#   function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
#            = conCat(con, sat, policy, m, s)
#
#  Example call: conCat(congp, gSat, policy, m, s)
#
# *Input arguments:*
#
#   con       function handle (controller)
#   sat       function handle (squashing function)
#   policy    policy structure (dict)
#     .maxU   maximum amplitude of control signal (after squashing)
#   m         mean of input distribution                             [D x 1]
#   s         covariance of input distribution                       [D x D]
#
# *Output arguments:*
#
#   M         control mean                                           [E   x   1]
#   S         control covariance                                     [E   x   E]
#   C         inv(s)*cov(x,u)                                        [D   x   E]
#   dMdm      deriv. of expected control wrt input mean              [E   x   D]
#   dSdm      deriv. of control covariance wrt input mean            [E*E x   D]
#   dCdm      deriv. of C wrt input mean                             [D*E x   D]
#   dMds      deriv. of expected control wrt input covariance        [E   x D*D]
#   dSds      deriv. of control covariance wrt input covariance      [E*E x D*D]
#   dCds      deriv. of C wrt input covariance                       [D*E x D*D]
#   dMdp      deriv. of expected control wrt policy parameters       [E   x   P]
#   dSdp      deriv. of control covariance wrt policy parameters     [E*E x   P]
#   dCdp      deriv. of C wrt policy parameters                      [D*E x   P]
#
#   where P is the total number of policy parameters
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2012-07-03
#
# High-Level Steps
# # Compute unsquashed control signal
# # Compute squashed control signal
#
# Translated from MATLAB 1-indexed to Python 0-indexed.

import numpy as np


def conCat(con, sat, policy, m, s, compute_derivatives=True):
    maxU = np.asarray(policy['maxU']).ravel()
    E = len(maxU)                         # dimension of control signal
    D = len(m)                            # dimension of input
    F = D + E

    # pre-compute some indices (0-based): j = control part, i = state part
    j = np.arange(D, F)                   # ndarray of length E
    i = np.arange(D)                      # ndarray of length D

    # initialize M and S
    M = np.zeros(F)
    M[i] = np.asarray(m).ravel()
    S = np.zeros((F, F))
    S[np.ix_(i, i)] = s

    if not compute_derivatives:
        # without derivatives
        M_j, S_jj, Q = con(policy, m, s, compute_derivatives=False)
        M[j] = M_j.ravel()
        S[np.ix_(j, j)] = S_jj
        q = s @ Q
        S[np.ix_(i, j)] = q
        S[np.ix_(j, i)] = q.T
        # compute squashed control signal
        # gSat returns M (size E), S (size ExE), R (size FxE)
        M, S_out, R = sat(M, S, j, maxU, compute_derivatives=False)
        C = np.hstack([np.eye(D), Q]) @ R
        return M, S_out, C

    # ------ with derivatives ------
    Mdm = np.zeros((F, D))
    Mdm[:D, :D] = np.eye(D)
    Mds = np.zeros((F, D * D))
    Sdm = np.zeros((F * F, D))
    Sds = np.kron(Mdm, Mdm)

    # Vectorized indices (column-major/Fortran order to match MATLAB)
    # Column-major linear index: idx(r, c) = r + c*F for (r,c) in [0,F-1]
    jj = (j[:, None] + j * F).ravel(order='F')    # j×j block
    ij = (i[:, None] + j * F).ravel(order='F')    # i×j block
    ji = (j[:, None] + i * F).ravel(order='F')    # j×i block

    # 1. Unsquashed controller --------------------------------------------------
    (M_j, S_jj, Q,
     dM_j_dm, dS_jj_dm, dQdm,
     dM_j_ds, dS_jj_ds, dQds,
     Mdp, Sdp, dQdp) = con(policy, m, s, compute_derivatives=True)

    M[j] = M_j.ravel()
    S[np.ix_(j, j)] = S_jj
    q = s @ Q
    S[np.ix_(i, j)] = q
    S[np.ix_(j, i)] = q.T

    # Fill in controller derivative contributions to Mdm, Mds
    Mdm[j, :] = dM_j_dm   # dM/dm for the control block
    Mds[j, :] = dM_j_ds   # dM/ds for the control block

    # update the derivatives
    SS = np.kron(np.eye(E), s)
    QQ = np.kron(Q.T, np.eye(D))
    Sdm[jj, :] = dS_jj_dm    # covariance derivative from controller (jj block)
    Sdm[ij, :] = SS @ dQdm
    Sdm[ji, :] = Sdm[ij, :]
    Sds[jj, :] = dS_jj_ds    # covariance derivative from controller (jj block)
    Sds[ij, :] = SS @ dQds + QQ
    Sds[ji, :] = Sds[ij, :]

    # 2. Apply Saturation -------------------------------------------------------
    # gSat returns: M (size E), S (size ExE), R (size FxE), plus derivatives
    (M_out, S_out, R,
     MdM, SdM, RdM,
     MdS, SdS, RdS) = sat(M, S, j, maxU, compute_derivatives=True)

    # apply chain-rule to compute derivatives after concatenation
    dMdm_out = MdM @ Mdm + MdS @ Sdm
    dMds_out = MdM @ Mds + MdS @ Sds
    dSdm_out = SdM @ Mdm + SdS @ Sdm
    dSds_out = SdM @ Mds + SdS @ Sds
    dRdm = RdM @ Mdm + RdS @ Sdm
    dRds = RdM @ Mds + RdS @ Sds

    dMdp = MdM[:, j] @ Mdp + MdS[:, jj] @ Sdp
    dSdp = SdM[:, j] @ Mdp + SdS[:, jj] @ Sdp
    dRdp = RdM[:, j] @ Mdp + RdS[:, jj] @ Sdp

    C = np.hstack([np.eye(D), Q]) @ R
    # update the derivatives
    RR = np.kron(R[j, :].T, np.eye(D))
    QQ2 = np.kron(np.eye(E), np.hstack([np.eye(D), Q]))
    dCdm = QQ2 @ dRdm + RR @ dQdm
    dCds = QQ2 @ dRds + RR @ dQds
    dCdp = QQ2 @ dRdp + RR @ dQdp

    return (M_out, S_out, C,
            dMdm_out, dSdm_out, dCdm,
            dMds_out, dSds_out, dCds,
            dMdp, dSdp, dCdp)
