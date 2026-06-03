# gTrig.py
# *Summary:* Compute moments of the saturating function e*sin(x(i)) and
# e*cos(x(i)), where x ~ N(m, v) and i is a (possibly empty)
# set of I indices. The optional scaling factor e is a vector of
# length I. Optionally, compute derivatives of the moments.
#
#    M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv = gTrig(m, v, i, e)
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
#   M     output means                                              [ 2I       ]
#   V     output covariance matrix                                  [ 2I x  2I ]
#   C     inv(v) times input-output covariance                      [ d  x  2I ]
#   dMdm  derivatives of M w.r.t m                                  [ 2I x   d ]
#   dVdm  derivatives of V w.r.t m                                  [4II x   d ]
#   dCdm  derivatives of C w.r.t m                                  [2dI x   d ]
#   dMdv  derivatives of M w.r.t v                                  [ 2I x d^2 ]
#   dVdv  derivatives of V w.r.t v                                  [4II x d^2 ]
#   dCdv  derivatives of C w.r.t v                                  [2dI x d^2 ]
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-25

import numpy as np


def gTrig(m, v, i, e=None, compute_derivatives=True):
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

    Is = 2 * np.arange(I)       # 0-based: [0, 2, 4, ...]
    Ic = Is + 1                 # 0-based: [1, 3, 5, ...]
    i_py = np.asarray(i).ravel().astype(int)

    if e is None:
        e = np.ones(I)
    else:
        e = np.asarray(e).ravel()
    ee = np.repeat(e, 2)        # [e0, e0, e1, e1, ...] length 2I

    mi = m[i_py]                # m(i) in MATLAB
    vi = v[np.ix_(i_py, i_py)]  # v(i,i) in MATLAB
    vii = np.diag(vi)           # diag(vi)

    M = np.zeros(2 * I)
    M[Is] = e * np.exp(-vii / 2) * np.sin(mi)
    M[Ic] = e * np.exp(-vii / 2) * np.cos(mi)

    lq = -(vii[:, np.newaxis] + vii[np.newaxis, :]) / 2
    q = np.exp(lq)
    mi_diff = mi[:, np.newaxis] - mi[np.newaxis, :]   # bsxfun(@minus, mi, mi')
    mi_sum = mi[:, np.newaxis] + mi[np.newaxis, :]     # bsxfun(@plus, mi, mi')

    U1 = (np.exp(lq + vi) - q) * np.sin(mi_diff)
    U2 = (np.exp(lq - vi) - q) * np.sin(mi_sum)
    U3 = (np.exp(lq + vi) - q) * np.cos(mi_diff)
    U4 = (np.exp(lq - vi) - q) * np.cos(mi_sum)

    V = np.zeros((2 * I, 2 * I))
    V[np.ix_(Is, Is)] = U3 - U4
    V[np.ix_(Ic, Ic)] = U3 + U4
    V[np.ix_(Is, Ic)] = U1 + U2
    V[np.ix_(Ic, Is)] = V[np.ix_(Is, Ic)].T
    V = np.outer(ee, ee) * V / 2

    C = np.zeros((d, 2 * I))
    C[np.ix_(i_py, Is)] = np.diag(M[Ic])
    C[np.ix_(i_py, Ic)] = np.diag(-M[Is])

    if not compute_derivatives:
        return M, V, C

    dVdm = np.zeros((2 * I, 2 * I, d))
    dCdm = np.zeros((d, 2 * I, d))
    dVdv = np.zeros((2 * I, 2 * I, d, d))
    dCdv = np.zeros((d, 2 * I, d, d))
    dMdm = C.T.copy()

    e_outer = np.outer(e, e)

    for j in range(I):
        u = np.zeros(I)
        u[j] = 0.5
        u_diff = u[:, np.newaxis] - u[np.newaxis, :]    # bsxfun(@minus, u, u')
        u_sum = u[:, np.newaxis] + u[np.newaxis, :]      # bsxfun(@plus, u, u')

        dVdm_slice = dVdm[:, :, i_py[j]]
        dVdm_slice[np.ix_(Is, Is)] = e_outer * (-U1 * u_diff + U2 * u_sum)
        dVdm_slice[np.ix_(Ic, Ic)] = e_outer * (-U1 * u_diff - U2 * u_sum)
        dVdm_slice[np.ix_(Is, Ic)] = e_outer * (U3 * u_diff + U4 * u_sum)
        dVdm_slice[np.ix_(Ic, Is)] = dVdm_slice[np.ix_(Is, Ic)].T

        dVdv[Is[j], Is[j], i_py[j], i_py[j]] = np.exp(-vii[j]) * \
            (1 + (2 * np.exp(-vii[j]) - 1) * np.cos(2 * mi[j])) * e[j] * e[j] / 2
        dVdv[Ic[j], Ic[j], i_py[j], i_py[j]] = np.exp(-vii[j]) * \
            (1 - (2 * np.exp(-vii[j]) - 1) * np.cos(2 * mi[j])) * e[j] * e[j] / 2
        dVdv[Is[j], Ic[j], i_py[j], i_py[j]] = np.exp(-vii[j]) * \
            (1 - 2 * np.exp(-vii[j])) * np.sin(2 * mi[j]) * e[j] * e[j] / 2
        dVdv[Ic[j], Is[j], i_py[j], i_py[j]] = dVdv[Is[j], Ic[j], i_py[j], i_py[j]]

        for k in list(range(j)) + list(range(j + 1, I)):
            dVdv[Is[j], Is[k], i_py[j], i_py[k]] = (
                np.exp(lq[j, k] + vi[j, k]) * np.cos(mi[j] - mi[k]) +
                np.exp(lq[j, k] - vi[j, k]) * np.cos(mi[j] + mi[k])
            ) * e[j] * e[k] / 2
            dVdv[Is[j], Is[k], i_py[j], i_py[j]] = -V[Is[j], Is[k]] / 2
            dVdv[Is[j], Is[k], i_py[k], i_py[k]] = -V[Is[j], Is[k]] / 2

            dVdv[Ic[j], Ic[k], i_py[j], i_py[k]] = (
                np.exp(lq[j, k] + vi[j, k]) * np.cos(mi[j] - mi[k]) -
                np.exp(lq[j, k] - vi[j, k]) * np.cos(mi[j] + mi[k])
            ) * e[j] * e[k] / 2
            dVdv[Ic[j], Ic[k], i_py[j], i_py[j]] = -V[Ic[j], Ic[k]] / 2
            dVdv[Ic[j], Ic[k], i_py[k], i_py[k]] = -V[Ic[j], Ic[k]] / 2

            dVdv[Ic[j], Is[k], i_py[j], i_py[k]] = -(
                np.exp(lq[j, k] + vi[j, k]) * np.sin(mi[j] - mi[k]) +
                np.exp(lq[j, k] - vi[j, k]) * np.sin(mi[j] + mi[k])
            ) * e[j] * e[k] / 2
            dVdv[Ic[j], Is[k], i_py[j], i_py[j]] = -V[Ic[j], Is[k]] / 2
            dVdv[Ic[j], Is[k], i_py[k], i_py[k]] = -V[Ic[j], Is[k]] / 2

            dVdv[Is[j], Ic[k], i_py[j], i_py[k]] = (
                np.exp(lq[j, k] + vi[j, k]) * np.sin(mi[j] - mi[k]) -
                np.exp(lq[j, k] - vi[j, k]) * np.sin(mi[j] + mi[k])
            ) * e[j] * e[k] / 2
            dVdv[Is[j], Ic[k], i_py[j], i_py[j]] = -V[Is[j], Ic[k]] / 2
            dVdv[Is[j], Ic[k], i_py[k], i_py[k]] = -V[Is[j], Ic[k]] / 2

        dCdm[i_py[j], Is[j], i_py[j]] = -M[Is[j]]
        dCdm[i_py[j], Ic[j], i_py[j]] = -M[Ic[j]]

        dCdv[i_py[j], Is[j], i_py[j], i_py[j]] = -C[i_py[j], Is[j]] / 2
        dCdv[i_py[j], Ic[j], i_py[j], i_py[j]] = -C[i_py[j], Ic[j]] / 2

    dMdv = np.transpose(dCdm, (1, 0, 2)) / 2

    dMdv = dMdv.reshape(2 * I, d * d, order='F')
    dVdv = dVdv.reshape(4 * I * I, d * d, order='F')
    dVdm = dVdm.reshape(4 * I * I, d, order='F')
    dCdv = dCdv.reshape(d * 2 * I, d * d, order='F')
    dCdm = dCdm.reshape(d * 2 * I, d, order='F')

    return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv
