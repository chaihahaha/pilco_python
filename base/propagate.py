# propagate.py
# *Summary:* Propagate the state distribution one time step forward.
#
#  [Mnext, Snext] = propagate(m, s, plant, dynmodel, policy)
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
#   Mnext             mean of the successor state at time t+1            [E x 1]
#   Snext             covariance of the successor state at time t+1      [E x E]
#
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


def propagate(m, s, plant, dynmodel, policy):
    ## Code

    m = np.asarray(m).ravel()
    s = np.atleast_2d(s)

    # extract important indices from structures
    angi = plant['angi']   # angular indices (0-based)
    poli = plant['poli']   # policy indices (0-based)
    dyni = plant['dyni']   # dynamics-model indices (0-based)
    difi = plant['difi']   # state indices where the model was trained on differences (0-based)

    D0 = len(m)                                        # size of the input mean
    D1 = D0 + 2 * len(angi)          # length after mapping all angles to sin/cos
    D2 = D1 + len(policy['maxU'])          # length after computing control signal
    D3 = D2 + D0                                         # length after predicting
    M = np.zeros(D3); M[:D0] = m; S = np.zeros((D3, D3)); S[:D0, :D0] = s   # init M and S

    # 1) Augment state distribution with trigonometric functions ------------------
    ii = np.arange(D0); jj = np.arange(D0); kk = np.arange(D0, D1)
    Mkk, Skk, C = gTrig(M[ii], S[np.ix_(ii, ii)], angi, compute_derivatives=False)
    M[kk] = Mkk
    S[np.ix_(kk, kk)] = Skk
    q = S[np.ix_(jj, ii)] @ C
    S[np.ix_(jj, kk)] = q
    S[np.ix_(kk, jj)] = q.T

    # sn2 = exp(2*dynmodel.hyp(end,:));  sn2(difi) = sn2(difi)/2;
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
    result3 = policy['fcn'](policy, mm[ii], ss[np.ix_(ii, ii)])
    Mkk, Skk, C3 = result3[0], result3[1], result3[2]
    M[kk] = Mkk
    S[np.ix_(kk, kk)] = Skk
    q3 = S[np.ix_(jj, ii)] @ C3
    S[np.ix_(jj, kk)] = q3
    S[np.ix_(kk, jj)] = q3.T

    # 3) Compute dynamics-GP prediction              ------------------------------
    ii = np.concatenate([dyni, np.arange(D1, D2)])
    jj = np.arange(D2)
    if 'sub' in dynmodel:
        Nf = len(dynmodel['sub'])
    else:
        Nf = 1
    for n in range(Nf):                               # potentially multiple dynamics models
        dyn, i_idx, k_idx = _slice_model(dynmodel, n, ii, D1, D2, D3)
        jj = np.setdiff1d(jj, k_idx, assume_unique=True)
        result = dyn['fcn'](dyn, M[i_idx], S[np.ix_(i_idx, i_idx)])
        Mk, Skk, C4 = result[0], result[1], result[2]
        M[k_idx] = Mk.ravel()
        S[np.ix_(k_idx, k_idx)] = Skk
        q4 = S[np.ix_(jj, i_idx)] @ C4
        S[np.ix_(jj, k_idx)] = q4
        S[np.ix_(k_idx, jj)] = q4.T

        jj = np.concatenate([jj, k_idx])                                   # update 'previous' state vector

    # 4) Compute distribution of the next state -----------------------------------
    P = np.hstack([np.zeros((D0, D2)), np.eye(D0)])
    if len(difi) > 0:
        P[np.ix_(difi, difi)] = np.eye(len(difi))
    Mnext = P @ M
    Snext = P @ S @ P.T
    Snext = (Snext + Snext.T) / 2

    return Mnext, Snext


def _slice_model(dynmodel, n, ii, D1, D2, D3):
    """Separate sub-dynamics models"""
    if 'sub' in dynmodel:
        dyn = dynmodel['sub'][n]
        do = dyn['dyno']   # output indices (0-based)
        D = len(ii) + D1 - D2
        di = dyn.get('dyni', np.array([], dtype=int))
        du = dyn.get('dynu', np.array([], dtype=int))
        dj = dyn.get('dynj', np.array([], dtype=int))
        i_idx = np.concatenate([ii[di], D1 + du, D2 + dj])
        k_idx = D2 + do
        # set inputs and targets for this sub-model
        input_cols2 = np.concatenate([di, D + du])
        dyn['inputs'] = np.column_stack([dynmodel['inputs'][:, input_cols2],
                                          dynmodel['targets'][:, dj]])
        dyn['target'] = dynmodel['targets'][:, do]
    else:
        dyn = dynmodel
        k_idx = np.arange(D2, D3)
        i_idx = ii
    return dyn, i_idx, k_idx
