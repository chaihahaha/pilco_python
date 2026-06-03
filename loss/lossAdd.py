# lossAdd.py
# *Summary:* Utility function to add a number of loss functions together, each of which
# can be using a different loss function and operating on a different part of
# the state.
#
#       def lossAdd(cost, m, s, compute_derivatives=True)
#
# *Input arguments:*
#
#  cost            cost dict
#     .fcn         lossAdd - called to arrive here
#     .sub         list of sub-loss dicts to add together
#        .fcn      handle to sub function
#        .losi     indices of variables to be passed to loss function (0-indexed)
#        .<>       all fields in sub will be passed onto the sub function
#     .expl        (optional) if present cost.expl*sqrt(S) is added to the loss
#
#   m         mean of input distribution                              [D x 1]
#   s         covariance matrix of input distribution                 [D x D]
#
# *Output arguments:*
#
#  L               expected loss                                  [1   x   1]
#  dLdm            derivative of L wrt input mean                 [1   x   D]
#  dLds            derivative of L wrt input covariance           [1   x   D^2]
#  S               variance of loss                               [1   x   1]
#  dSdm            derivative of S wrt input mean                 [1   x   D]
#  dSds            derivative of S wrt input covariance           [1   x   D^2]
#  C               inv(S) times input-output covariance           [D   x   1]
#  dCdm            derivative of C wrt input mean                 [D   x   D]
#  dCds            derivative of C wrt input covariance           [D   x   D^2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-05

import numpy as np


def _sub2ind2(D, i, j):
    # D = #rows, i = row subscript, j = column subscript
    i = i.ravel()
    j = j.ravel()
    # bsxfun(@plus, D*(j-1), i)  -- MATLAB 1-indexed, column-major
    # For Python 0-indexed: idx = i + j * D
    idx = i[:, np.newaxis] + j[np.newaxis, :] * D
    return idx.ravel()


def lossAdd(cost, m_input, s_input, compute_derivatives=True):
    ## Code

    # Dimensions and Initializations
    Nlos = len(cost['sub'])
    D = len(m_input)

    m = np.atleast_2d(m_input).reshape(-1, 1)
    s = np.atleast_2d(s_input)

    L = 0.0
    S_var = 0.0
    C = np.zeros((D, 1))
    dLdm = np.zeros((1, D))
    dSdm = np.zeros((1, D))
    dLds = np.zeros((D, D))
    dSds = np.zeros((D, D))
    dCdm = np.zeros((D, D))
    dCds = np.zeros((D, D * D))

    for n in range(Nlos):
        costi = cost['sub'][n]
        i = np.atleast_1d(costi['losi']).ravel().astype(int)
        i_ix = np.ix_(i, i)

        if not compute_derivatives:
            # Just the expected loss & derivs
            Li, Ldm, Lds, *_ = costi['fcn'](costi, m[i], s[i_ix], compute_derivatives=False)

            L = L + Li
            dLdm[0, i] = dLdm[0, i] + Ldm.ravel()
            dLds[i_ix] = dLds[i_ix] + Lds

        else:
            # Also loss variance and IO covariance
            result = costi['fcn'](costi, m[i], s[i_ix], compute_derivatives=True)
            Li, Ldm, Lds, Si, Sdm, Sds, Ci, Cdm, Cds = result

            L = L + Li
            # V(a+b) = V(a)+V(b)+C(a,b)+C(b,a)
            S_var = S_var + Si + (Ci.T @ s[i, :] @ C + C.T @ s[:, i] @ Ci).item()

            # derivatives
            dLdm[0, i] = dLdm[0, i] + Ldm.ravel()
            dLds[i_ix] = dLds[i_ix] + Lds

            Cis = Ci.T @ (s[i, :] + s[:, i].T)
            Cs = C.T @ (s[:, i] + s[i, :].T)

            dSdm[0, i] = dSdm[0, i] + Sdm.ravel() + (Cs @ Cdm).ravel()
            dSdm = dSdm + Cis @ dCdm

            dSds[i_ix] = dSds[i_ix] + Sds + (Cs @ Cds).reshape(len(i), len(i))
            dSds = dSds + (Cis @ dCds).reshape(D, D)

            # Cross terms between sub-loss and rest
            dSds[i, :] = dSds[i, :] + Ci @ C.T
            dSds[:, i] = dSds[:, i] + C @ Ci.T

            # Input - Output covariance update
            C[i] = C[i] + Ci  # must be after S and its derivatives

            ii = _sub2ind2(D, i, i)
            # must be after dSdm & dSds
            dCdm[np.ix_(i, i)] = dCdm[np.ix_(i, i)] + Cdm
            dCds[np.ix_(i, ii)] = dCds[np.ix_(i, ii)] + Cds

    # Exploration if required
    if 'expl' in cost and compute_derivatives and cost['expl'] != 0:
        L = L + cost['expl'] * np.sqrt(S_var)
        dLdm = dLdm + cost['expl'] * 0.5 / np.sqrt(S_var) * dSdm
        dLds = dLds + cost['expl'] * 0.5 / np.sqrt(S_var) * dSds

    return L, dLdm, dLds, S_var, dSdm, dSds, C, dCdm, dCds
