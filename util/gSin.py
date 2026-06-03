# gSin.py
# *Summary:* Compute moments of the saturating function e*sin(x(i)),
# where x ~ N(m,v) and i is a (possibly empty) set of I
# indices. The optional  scaling factor e is a vector of length I.
# Optionally, compute derivatives of the moments.
#
#    [M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = gSin(m, v, i, e, compute_derivatives=True)
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


def gSin(m, v, i, e=None, compute_derivatives=True):
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

    mi = m[i_py]
    vi = v[np.ix_(i_py, i_py)]
    vii = np.diag(vi)

    M = e * np.exp(-vii / 2) * np.sin(mi)

    lq = -(vii[:, None] + vii[None, :]) / 2
    q = np.exp(lq)
    mi_diff = mi[:, None] - mi[None, :]
    mi_sum = mi[:, None] + mi[None, :]

    V = (np.exp(lq + vi) - q) * np.cos(mi_diff) - \
        (np.exp(lq - vi) - q) * np.cos(mi_sum)
    V = np.outer(e, e) * V / 2

    C = np.zeros((d, I))
    C[np.ix_(i_py, np.arange(I))] = np.diag(e * np.exp(-vii / 2) * np.cos(mi))

    if not compute_derivatives:
        return M, V, C

    dVdm = np.zeros((I, I, d))
    dCdm = np.zeros((d, I, d))
    dVdv = np.zeros((I, I, d, d))
    dCdv = np.zeros((d, I, d, d))
    dMdm = C.T.copy()

    U1 = -(np.exp(lq + vi) - q) * np.sin(mi_diff)
    U2 = (np.exp(lq - vi) - q) * np.sin(mi_sum)

    e_outer = np.outer(e, e)

    for j in range(I):
        u = np.zeros(I)
        u[j] = 0.5
        u_diff = u[:, None] - u[None, :]
        u_sum = u[:, None] + u[None, :]

        dVdm[:, :, i_py[j]] = e_outer * (U1 * u_diff + U2 * u_sum)

        dVdv[j, j, i_py[j], i_py[j]] = np.exp(-vii[j]) * \
            (1 + (2 * np.exp(-vii[j]) - 1) * np.cos(2 * mi[j])) * e[j] * e[j] / 2

        for k in list(range(j)) + list(range(j + 1, I)):
            dVdv[j, k, i_py[j], i_py[k]] = (
                np.exp(lq[j, k] + vi[j, k]) * np.cos(mi[j] - mi[k]) +
                np.exp(lq[j, k] - vi[j, k]) * np.cos(mi[j] + mi[k])
            ) * e[j] * e[k] / 2
            dVdv[j, k, i_py[j], i_py[j]] = -V[j, k] / 2
            dVdv[j, k, i_py[k], i_py[k]] = -V[j, k] / 2

        dCdm[i_py[j], j, i_py[j]] = -M[j]
        dCdv[i_py[j], j, i_py[j], i_py[j]] = -C[i_py[j], j] / 2

    dMdv = np.transpose(dCdm, (1, 0, 2)) / 2

    dMdv = dMdv.reshape(I, d * d, order='F')
    dVdv = dVdv.reshape(I * I, d * d, order='F')
    dVdm = dVdm.reshape(I * I, d, order='F')
    dCdv = dCdv.reshape(d * I, d * d, order='F')
    dCdm = dCdm.reshape(d * I, d, order='F')

    return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv
