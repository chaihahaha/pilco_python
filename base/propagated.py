# propagated.py
# *Summary:* Propagate the state distribution one time step forward
#            with derivatives
#
#  function [Mnext, Snext, dMdm, dSdm, dMds, dSds, dMdp, dSdp] = ...
#    propagated(m, s, plant, dynmodel, policy)
#
# *Input arguments:*
#
#   m                 mean of the state distribution at time t           [D x 1]
#   s                 covariance of the state distribution at time t     [D x D]
#   plant             plant structure
#   dynmodel          dynamics model structure
#   policy            policy structure
#
# *Output arguments:*
#
#   Mnext             predicted mean at time t+1                         [E x 1]
#   Snext             predicted covariance at time t+1                   [E x E]
#   dMdm              output mean wrt input mean                         [E x D]
#   dMds              output mean wrt input covariance matrix         [E  x D*D]
#   dSdm              output covariance matrix wrt input mean        [E*E x  D ]
#   dSds              output cov wrt input cov                       [E*E x D*D]
#   dMdp              output mean wrt policy parameters                  [E x P]
#   dSdp              output covariance matrix wrt policy parameters  [E*E x  P]
#
#   where P is the number of policy parameters.
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, Henrik Ohlsson,
# and Carl Edward Rasmussen.
#
# Last modified: 2016-07-19
#
# High-Level Steps
# # Augment state distribution with trigonometric functions
# # Compute distribution of the control signal
# # Compute dynamics-GP prediction
# # Compute distribution of the next state


import numpy as np
from pilco_python.util.gTrig import gTrig
from .propagate import propagate


def propagated(m, s, plant, dynmodel, policy, compute_derivatives=False):
    ## Code

    m = np.asarray(m).ravel()
    s = np.atleast_2d(s)

    if not compute_derivatives:                # just predict, no derivatives
        return propagate(m, s, plant, dynmodel, policy)

    angi = plant['angi']
    poli = plant['poli']
    dyni = plant['dyni']
    difi = plant['difi']

    D0 = len(m)                                        # size of the input mean
    D1 = D0 + 2 * len(angi)          # length after mapping all angles to sin/cos
    D2 = D1 + len(policy['maxU'])          # length after computing control signal
    D3 = D2 + D0                                         # length after predicting
    M = np.zeros(D3); M[:D0] = m; S = np.zeros((D3, D3)); S[:D0, :D0] = s   # init M and S

    Mdm = np.vstack([np.eye(D0), np.zeros((D3 - D0, D0))])
    Sdm = np.zeros((D3 * D3, D0))
    Mds = np.zeros((D3, D0 * D0))
    Sds = np.kron(Mdm, Mdm)
    X = np.arange(1, D3 * D3 + 1).reshape((D3, D3), order='F')
    XT = X.T.copy()
    Sds = (Sds + Sds[XT.ravel(order='F').astype(int) - 1, :]) / 2
    X = np.arange(1, D0 * D0 + 1).reshape((D0, D0), order='F')
    XT = X.T.copy()
    Sds = (Sds + Sds[:, XT.ravel(order='F').astype(int) - 1]) / 2

    # 1) Augment state distribution with trigonometric functions ------------------
    ii = np.arange(D0); jj = np.arange(D0); kk = np.arange(D0, D1)
    Mkk, Skk, C, mdm, sdm, Cdm, mds, sds, Cds = gTrig(M[ii], S[np.ix_(ii, ii)], angi, compute_derivatives=True)
    M[kk] = Mkk
    S[np.ix_(kk, kk)] = Skk
    S, Mdm, Mds, Sdm, Sds, Mdp, Sdp = _fill_in(
        S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
        None, None, None, ii, jj, kk, D3)

    # sn2 = exp(2*dynmodel.hyp(end,:)); sn2(difi) = sn2(difi)/2;
    mm = np.zeros(D1); mm[ii] = M[ii]
    ss = np.zeros((D1, D1)); ss[np.ix_(ii, ii)] = S[np.ix_(ii, ii)]  # +diag(sn2);
    mmkk, sskk, C2 = gTrig(mm[ii], ss[np.ix_(ii, ii)], angi, compute_derivatives=False)
    mm[kk] = mmkk
    ss[np.ix_(kk, kk)] = sskk
    q2 = ss[np.ix_(jj, ii)] @ C2
    ss[np.ix_(jj, kk)] = q2
    ss[np.ix_(kk, jj)] = q2.T

    # 2) Compute distribution of the control signal -------------------------------
    ii = poli; jj = np.arange(D1); kk = np.arange(D1, D2)
    Mkk, Skk, C, mdm, sdm, Cdm, mds, sds, Cds, Mdp, Sdp, Cdp = \
        policy['fcn'](policy, mm[ii], ss[np.ix_(ii, ii)], compute_derivatives=True)
    M[kk] = Mkk
    S[np.ix_(kk, kk)] = Skk
    S, Mdm, Mds, Sdm, Sds, Mdp, Sdp = _fill_in(
        S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
        Mdp, Sdp, Cdp, ii, jj, kk, D3)

    # 3) Compute distribution of the change in state ------------------------------
    ii = np.concatenate([dyni, np.arange(D1, D2)])
    jj = np.arange(D2)
    if 'sub' in dynmodel:
        Nf = len(dynmodel['sub'])
    else:
        Nf = 1
    for n in range(Nf):                               # potentially multiple dynamics models
        dyn, i_idx, k_idx = _slice_model(dynmodel, n, ii, D1, D2, D3)
        jj = np.setdiff1d(jj, k_idx, assume_unique=True)

        Mk, Skk, C, mdm, sdm, Cdm, mds, sds, Cds = \
            dyn['fcn'](dyn, M[i_idx], S[np.ix_(i_idx, i_idx)], compute_derivatives=True)
        M[k_idx] = Mk.ravel()
        S[np.ix_(k_idx, k_idx)] = Skk
        S, Mdm, Mds, Sdm, Sds, Mdp, Sdp = _fill_in(
            S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
            Mdp, Sdp, None, i_idx, jj, k_idx, D3)

        jj = np.concatenate([jj, k_idx])                                   # update 'previous' state vector

    # 4) Compute distribution of the next state -----------------------------------
    P = np.hstack([np.zeros((D0, D2)), np.eye(D0)])
    if len(difi) > 0:
        P[np.ix_(difi, difi)] = np.eye(len(difi))
    # P = sparse(P) in MATLAB, but we keep it dense
    Mnext = P @ M
    Snext = P @ S @ P.T
    Snext = (Snext + Snext.T) / 2

    PP = np.kron(P, P)
    dMdm = P @ Mdm
    dMds = P @ Mds
    dMdp = P @ Mdp
    dSdm = PP @ Sdm
    dSds = PP @ Sds
    dSdp = PP @ Sdp

    X = np.arange(1, D0 * D0 + 1).reshape((D0, D0), order='F')
    XT = X.T.copy()
    xt_vec = XT.ravel(order='F').astype(int) - 1                           # symmetrize dS
    dSdm = (dSdm + dSdm[xt_vec, :]) / 2
    dMds = (dMds + dMds[:, xt_vec]) / 2
    dSds = (dSds + dSds[xt_vec, :]) / 2
    dSds = (dSds + dSds[:, xt_vec]) / 2
    dSdp = (dSdp + dSdp[xt_vec, :]) / 2

    return Mnext, Snext, dMdm, dSdm, dMds, dSds, dMdp, dSdp


def _slice_model(dynmodel, n, ii, D1, D2, D3):
    """Separate sub-dynamics models"""
    if 'sub' in dynmodel:
        dyn = dynmodel['sub'][n]
        do = dyn['dyno']
        D = len(ii) + D1 - D2
        di = dyn.get('dyni', np.array([], dtype=int))
        du = dyn.get('dynu', np.array([], dtype=int))
        dj = dyn.get('dynj', np.array([], dtype=int))
        i_idx = np.concatenate([ii[di], D1 + du, D2 + dj])
        k_idx = D2 + do
        input_cols2 = np.concatenate([di, D + du])
        dyn['inputs'] = np.column_stack([dynmodel['inputs'][:, input_cols2],
                                          dynmodel['targets'][:, dj]])
        dyn['target'] = dynmodel['targets'][:, do]
    else:
        dyn = dynmodel
        k_idx = np.arange(D2, D3)
        i_idx = ii
    return dyn, i_idx, k_idx


def _fill_in(S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds,
             Mdp, Sdp, dCdp, i, j, k, D):
    """Apply chain rule and fill out cross covariance terms"""

    if len(k) == 0:
        return S, Mdm, Mds, Sdm, Sds, Mdp, Sdp

    # vectorized indices (Fortran order, 1-based like MATLAB)
    X = np.arange(1, D * D + 1).reshape((D, D), order='F')
    XT = X.T.copy()

    I_mat = np.zeros((D, D))
    I_mat[np.ix_(i, i)] = 1
    ii_vec = X[I_mat == 1]          # This is C-order flattening; need to fix

    # Use Fortran-order flattening for logical indexing
    I_flat = I_mat.ravel(order='F')
    ii_vec = X.ravel(order='F')[I_flat == 1]

    I_mat = np.zeros((D, D))
    I_mat[np.ix_(k, k)] = 1
    I_flat = I_mat.ravel(order='F')
    kk_vec = X.ravel(order='F')[I_flat == 1]

    I_mat = np.zeros((D, D))
    I_mat[np.ix_(j, i)] = 1
    I_flat = I_mat.ravel(order='F')
    ji_vec = X.ravel(order='F')[I_flat == 1]

    I_mat = np.zeros((D, D))
    I_mat[np.ix_(j, k)] = 1
    I_flat = I_mat.ravel(order='F')
    jk_vec = X.ravel(order='F')[I_flat == 1]

    I_mat = np.zeros((D, D))
    I_mat[np.ix_(j, k)] = 1
    I_flat = I_mat.ravel(order='F')
    kj_vec = XT.ravel(order='F')[I_flat == 1]

    # Convert to 0-based
    ii_idx = ii_vec.astype(int) - 1
    kk_idx = kk_vec.astype(int) - 1
    ji_idx = ji_vec.astype(int) - 1
    jk_idx = jk_vec.astype(int) - 1
    kj_idx = kj_vec.astype(int) - 1

    # chain rule
    Mdm[k, :] = mdm @ Mdm[i, :] + mds @ Sdm[ii_idx, :]
    Mds[k, :] = mdm @ Mds[i, :] + mds @ Sds[ii_idx, :]
    Sdm[kk_idx, :] = sdm @ Mdm[i, :] + sds @ Sdm[ii_idx, :]
    Sds[kk_idx, :] = sdm @ Mds[i, :] + sds @ Sds[ii_idx, :]
    dCdm = Cdm @ Mdm[i, :] + Cds @ Sdm[ii_idx, :]
    dCds = Cdm @ Mds[i, :] + Cds @ Sds[ii_idx, :]

    if dCdp is None and Mdp is not None:
        Mdp[k, :] = mdm @ Mdp[i, :] + mds @ Sdp[ii_idx, :]
        Sdp[kk_idx, :] = sdm @ Mdp[i, :] + sds @ Sdp[ii_idx, :]
        dCdp = Cdm @ Mdp[i, :] + Cds @ Sdp[ii_idx, :]
    elif Mdp is not None:
        aa = len(k)
        bb = aa * aa
        cc = C.size  # numel(C)
        mdp = np.zeros((D, Mdp.shape[1]))
        sdp = np.zeros((D * D, Mdp.shape[1]))
        mdp[k, :] = Mdp.reshape(aa, -1)
        Mdp = mdp
        sdp[kk_idx, :] = Sdp.reshape(bb, -1)
        Sdp = sdp
        Cdp = dCdp.reshape(cc, -1)
        dCdp = Cdp

    # off-diagonal
    q = S[np.ix_(j, i)] @ C
    S[np.ix_(j, k)] = q
    S[np.ix_(k, j)] = q.T

    SS = np.kron(np.eye(len(k)), S[np.ix_(j, i)])
    CC = np.kron(C.T, np.eye(len(j)))
    Sdm[jk_idx, :] = SS @ dCdm + CC @ Sdm[ji_idx, :]
    Sdm[kj_idx, :] = Sdm[jk_idx, :]
    Sds[jk_idx, :] = SS @ dCds + CC @ Sds[ji_idx, :]
    Sds[kj_idx, :] = Sds[jk_idx, :]
    if Mdp is not None:
        Sdp[jk_idx, :] = SS @ dCdp + CC @ Sdp[ji_idx, :]
        Sdp[kj_idx, :] = Sdp[jk_idx, :]

    return S, Mdm, Mds, Sdm, Sds, Mdp, Sdp
