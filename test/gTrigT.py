# gTrigT.py
# *Summary:* Test the gTrig function, which computes (at least) the mean and
# the variance of the transformed variable for a Gaussian distributed input
# x~N(m,v). Check the outputs using Monte Carlo, and the
# derivatives using finite differences.
#
#
#   gTrigT(m, v, i, e)
#
#
# *Input arguments:*
#
#   m     mean vector of Gaussian                                    [ d       ]
#   v     covariance matrix                                          [ d  x  d ]
#   i     vector of indices of elements to augment (0-based)         [ I       ]
#   e     (optional) scale vector; default: 1                        [ I       ]
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-25

import numpy as np

from pilco_python.util.gTrig import gTrig
from pilco_python.test.checkgrad import checkgrad


def gTrigT(m=None, v=None, i=None, e=None):
    # create a default test if no input arguments are given
    if m is None:
        np.random.seed(42)
        D = 4
        m = np.array([1.2, -0.5, 0.8, -1.1])
        v = np.eye(D) + np.random.randn(D, D); v = v @ v.T
        i = np.array([1, 3], dtype=int)  # 0-based, corresponds to [2, 4] in MATLAB 1-based
        I = 2 * len(i)
        e = np.array([2.3, 1.7])
    else:
        D = len(m)

    n = int(1e4)                                         # Monte Carlo sample size
    delta = 1e-4                                         # for finite difference approx

    x = m.reshape(-1, 1) + np.linalg.cholesky(v).T @ np.random.randn(D, n)
    ei = np.concatenate([e, e])
    y_sin = np.sin(x[i, :])
    y_cos = np.cos(x[i, :])
    y = ei.reshape(-1, 1) * np.vstack([y_sin, y_cos])
    ii = len(i)
    reorder = np.concatenate([np.arange(ii), ii + np.arange(ii)])
    y = y[reorder, :]

    M, V, C = gTrig(m, v, i, e, compute_derivatives=False)
    Q = np.cov(np.vstack([x, y]))
    Qv = Q[D:, D:]
    Qc = np.linalg.solve(v, Q[:D, D:])

    print('mean: gTrig Monte Carlo')
    print(np.column_stack([M.ravel(), np.mean(y, axis=1)]))
    print(' ')

    print('var:  gTrig Monte Carlo')
    print(np.column_stack([V.ravel(), Qv.ravel()]))
    print(' ')

    print('cov:  gTrig Monte Carlo')
    print(np.column_stack([C.ravel(order='F'), Qc.ravel(order='F')]))
    print(' ')

    I = len(i) * 2  # total number of augmented dimensions

    print('dMdm')
    for j in range(I):
        checkgrad(lambda x, jj=j: _gTrigT0(x, v, i, e, jj), m, delta)
        print(f'this was element # {j + 1}/{I}')
    print(' ')

    print('dVdm')
    for j in range(I * I):
        checkgrad(lambda x, jj=j: _gTrigT1(x, v, i, e, jj), m, delta)
        print(f'this was element # {j + 1}/{I * I}')
    print(' ')

    print('dCdm')
    for j in range(I * D):
        checkgrad(lambda x, jj=j: _gTrigT2(x, v, i, e, jj), m, delta)
        print(f'this was element # {j + 1}/{I * D}')
    print(' ')

    print('dMdv')
    for j in range(I):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gTrigT3(x, m, i, e, jj), v_lower, delta)
        print(f'this was element # {j + 1}/{I}')
    print(' ')

    print('dVdv')
    for j in range(I * I):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gTrigT4(x, m, i, e, jj), v_lower, delta)
        print(f'this was element # {j + 1}/{I * I}')
    print(' ')

    print('dCdv')
    for j in range(I * D):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gTrigT5(x, m, i, e, jj), v_lower, delta)
        print(f'this was element # {j + 1}/{I * D}')


def _gTrigT0(m, v, i, e, j):
    M, V, C, dMdm = gTrig(m, v, i, e, compute_derivatives=True)[:4]
    f = M[j]
    df = dMdm[j, :]
    return f, df


def _gTrigT1(m, v, i, e, j):
    M, V, C, dMdm, dVdm = gTrig(m, v, i, e, compute_derivatives=True)[:5]
    dVdm = dVdm.reshape(V.shape[0], V.shape[1], len(m), order='F')
    dd = V.shape[0]
    p = j // dd
    q = j % dd
    f = V[p, q]
    df = dVdm[p, q, :]
    return f, df


def _gTrigT2(m, v, i, e, j):
    M, V, C, dMdm, dVdm, dCdm = gTrig(m, v, i, e, compute_derivatives=True)[:6]
    dCdm = dCdm.reshape(C.shape[0], C.shape[1], len(m), order='F')
    dd = C.shape[0]
    p = j // dd
    q = j % dd
    f = C[p, q]
    df = dCdm[p, q, :]
    return f, df


def _gTrigT3(v, m, i, e, j):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm, dVdm, dCdm, dMdv = gTrig(m, vv, i, e, compute_derivatives=True)[:7]
    dMdv = dMdv.reshape(len(M), d, d, order='F')
    f = M[j]
    df = dMdv[j, :, :]
    df = df + df.T - np.diag(np.diag(df))
    idx = np.tril_indices(d)
    df = df[idx]
    return f, df


def _gTrigT4(v, m, i, e, j):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv = gTrig(m, vv, i, e, compute_derivatives=True)[:8]
    dVdv = dVdv.reshape(V.shape[0], V.shape[1], d, d, order='F')
    dd = V.shape[0]
    p = j // dd
    q = j % dd
    f = V[p, q]
    df = dVdv[p, q, :, :]
    df = df + df.T - np.diag(np.diag(df))
    idx = np.tril_indices(d)
    df = df[idx]
    return f, df


def _gTrigT5(v, m, i, e, j):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv = gTrig(m, vv, i, e, compute_derivatives=True)
    dCdv = dCdv.reshape(C.shape[0], C.shape[1], d, d, order='F')
    dd = C.shape[0]
    p = j // dd
    q = j % dd
    f = C[p, q]
    df = dCdv[p, q, :, :]
    df = df + df.T - np.diag(np.diag(df))
    idx = np.tril_indices(d)
    df = df[idx]
    return f, df
