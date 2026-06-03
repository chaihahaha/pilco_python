# gSat.py
# *Summary:* Compute moments of the saturating function
# e*(9*sin(x(i))+sin(3*x(i)))/8,
# where x ~ N(m,v) and i is a (possibly empty) set of I
# indices. The optional  scaling factor e is a vector of length I.
# Optionally, compute derivatives of the moments.
#
#    [M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = gSat(m, v, i, e, compute_derivatives=True)
#
# *Input arguments:*
#
#   m     mean vector of Gaussian                                    [ d       ]
#   v     covariance matrix                                          [ d  x  d ]
#   i     vector of indices of elements to augment (0-based)         [ I       ]
#   e     (optional) scale vector; default: 1                        [ I       ]
#
# *Output arguments:*
#
#   M     output means                                               [ I       ]
#   V     output covariance matrix                                   [ I  x  I ]
#   C     inv(v) times input-output covariance                       [ d  x  I ]
#   dMdm  derivatives of M w.r.t m                                   [ I  x  d ]
#   dVdm  derivatives of V w.r.t m                                   [I^2 x  d ]
#   dCdm  derivatives of C w.r.t m                                   [d*I x  d ]
#   dMdv  derivatives of M w.r.t v                                   [ I  x d^2]
#   dVdv  derivatives of V w.r.t v                                   [I^2 x d^2]
#   dCdv  derivatives of C w.r.t v                                   [d*I x d^2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-25

import numpy as np
try:
    from .gSin import gSin
except ImportError:
    from util.gSin import gSin


def gSat(m, v, i, e=None, compute_derivatives=True):
    ## Code
    d = len(m)
    I = len(i)

    if I == 0:
        M = np.zeros(0)
        V = np.zeros((0, 0))
        C = np.zeros((d, 0))
        if compute_derivatives:
            return M, V, C, np.zeros((0, d)), np.zeros((0, d)), np.zeros((0, d)), \
                   np.zeros((0, d*d)), np.zeros((0, d*d)), np.zeros((0, d*d))
        return M, V, C

    i_py = np.asarray(i).ravel().astype(int)

    if e is None:
        e = np.ones(I)
    else:
        e = np.asarray(e).ravel()

    # augment inputs
    P = np.vstack([np.eye(d), 3 * np.eye(d)])
    ma = np.concatenate([m, 3 * m])
    madm = P.copy()
    va = P @ v @ P.T
    vadv = np.kron(P, P)
    va = (va + va.T) / 2

    # do the actual augmentation with the right parameters
    M2, S2, C2, Mdma, Sdma, Cdma, Mdva, Sdva, Cdva = \
        gSin(ma, va, np.concatenate([i_py, d + i_py]),
             np.concatenate([9 * e, e]) / 8, compute_derivatives=True)

    P_comb = np.hstack([np.eye(I), np.eye(I)])
    Q_comb = np.hstack([np.eye(d), 3 * np.eye(d)])
    M = P_comb @ M2
    V = P_comb @ S2 @ P_comb.T
    V = (V + V.T) / 2
    C = Q_comb @ C2 @ P_comb.T

    if not compute_derivatives:
        return M, V, C

    dMdm = P_comb @ Mdma @ madm
    dMdv = P_comb @ Mdva @ vadv
    dVdm = np.kron(P_comb, P_comb) @ Sdma @ madm
    dVdv = np.kron(P_comb, P_comb) @ Sdva @ vadv
    dCdm = np.kron(P_comb, Q_comb) @ Cdma @ madm
    dCdv = np.kron(P_comb, Q_comb) @ Cdva @ vadv

    return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv
