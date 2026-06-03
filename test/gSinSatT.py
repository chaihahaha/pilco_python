# gSinSatT.py
# *Summary:* Test the gSin and gSat functions.
# Check the predictions using Monte Carlo and the derivatives by
# finite differences.
#
#
#   gSinSatT(fcn, m, v, i, e)
#
#
# *Input arguments:*
#
#   fcn      'gSin' or 'gSat'
#   m         mean of input distribution                         [D x 1]
#   v         covariance matrix of input distribution            [D x D]
#   i         vector of indices of elements to augment (0-based) [I]
#   e         (optional) scaling vector; default: 1              [I]
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-21

import numpy as np

from pilco_python.util.gSin import gSin
from pilco_python.util.gSat import gSat
from pilco_python.test.checkgrad import checkgrad


def gSinSatT(fcn='gSin', m=None, v=None, i=None, e=None):
    if m is None:
        np.random.seed(42)
        D = 4
        m = np.array([1.2, -0.5, 0.8, -1.1])
        r = np.random.randn(D, D)
        v = np.eye(D) + r @ r.T
        i = np.array([0, 1, 3], dtype=int)  # 0-based: [1, 2, 4] in MATLAB 1-based
        I = len(i)
        e = np.array([1.8, 2.5, 0.9])
    else:
        D = len(m)
        I = len(i)

    if fcn not in ('gSin', 'gSat'):
        raise ValueError('Can only handle gSin and gSat')

    n = int(1e4)                                         # Monte Carlo sample size
    delta = 1e-4                                         # for finite difference approx

    x = m.reshape(-1, 1) + np.linalg.cholesky(v).T @ np.random.randn(D, n)

    if fcn == 'gSin':
        y = e.reshape(-1, 1) * np.sin(x[i, :])
    else:
        y = e.reshape(-1, 1) * (9 * np.sin(x[i, :]) / 8 + np.sin(3 * x[i, :]) / 8)

    fn = gSin if fcn == 'gSin' else gSat

    M, V, C = fn(m, v, i, e, compute_derivatives=False)
    Q = np.cov(np.vstack([x, y]))
    Qv = Q[D:, D:]
    Qc = np.linalg.solve(v, Q[:D, D:])

    print(f'mean: {fcn}  Monte Carlo')
    print(np.column_stack([M.ravel(), np.mean(y, axis=1)]))
    print(' ')

    print(f'var:  {fcn}  Monte Carlo')
    print(np.column_stack([V.ravel(order='F'), Qv.ravel(order='F')]))
    print(' ')

    print(f'cov:  {fcn}  Monte Carlo')
    print(np.column_stack([C.ravel(order='F'), Qc.ravel(order='F')]))
    print(' ')

    print('dMdm')
    for j in range(I):
        checkgrad(lambda x, jj=j: _gSinT0(x, v, i, e, jj, fn), m, delta)
        print(f'this was element # {j + 1}/{I}')
    print(' ')

    print('dVdm')
    for j in range(I * I):
        checkgrad(lambda x, jj=j: _gSinT1(x, v, i, e, jj, fn), m, delta)
        print(f'this was element # {j + 1}/{I * I}')
    print(' ')

    print('dCdm')
    for j in range(I * D):
        checkgrad(lambda x, jj=j: _gSinT2(x, v, i, e, jj, fn), m, delta)
        print(f'this was element # {j + 1}/{I * D}')
    print(' ')

    print('dMdv')
    for j in range(I):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gSinT3(x, m, i, e, jj, fn), v_lower, delta)
        print(f'this was element # {j + 1}/{I}')
    print(' ')

    print('dVdv')
    for j in range(I * I):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gSinT4(x, m, i, e, jj, fn), v_lower, delta)
        print(f'this was element # {j + 1}/{I * I}')
    print(' ')

    print('dCdv')
    for j in range(I * D):
        idx = np.tril_indices_from(v)
        v_lower = v[idx]
        checkgrad(lambda x, jj=j: _gSinT5(x, m, i, e, jj, fn), v_lower, delta)
        print(f'this was element # {j + 1}/{I * D}')


def _gSinT0(m, v, i, e, j, fn):
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm = fn(m, v, i, e, compute_derivatives=True)[:4]
    f = M[j]
    df = dMdm[j, :]
    return f, df


def _gSinT1(m, v, i, e, j, fn):
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm, dVdm = fn(m, v, i, e, compute_derivatives=True)[:5]
    dVdm = dVdm.reshape(V.shape[0], V.shape[1], len(m), order='F')
    dd = V.shape[0]
    p = j // dd
    q = j % dd
    f = V[p, q]
    df = dVdm[p, q, :]
    return f, df


def _gSinT2(m, v, i, e, j, fn):
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    M, V, C, dMdm, dVdm, dCdm = fn(m, v, i, e, compute_derivatives=True)[:6]
    dCdm = dCdm.reshape(C.shape[0], C.shape[1], len(m), order='F')
    dd = C.shape[0]
    p = j // dd
    q = j % dd
    f = C[p, q]
    df = dCdm[p, q, :]
    return f, df


def _gSinT3(v, m, i, e, j, fn):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    out = fn(m, vv, i, e, compute_derivatives=True)
    M = out[0]
    dMdv = out[6]
    dMdv = dMdv.reshape(len(M), d, d, order='F')
    f = M[j]
    df = dMdv[j, :, :]
    df = df + df.T - np.diag(np.diag(df))
    idx = np.tril_indices(d)
    df = df[idx]
    return f, df


def _gSinT4(v, m, i, e, j, fn):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    out = fn(m, vv, i, e, compute_derivatives=True)
    M = out[0]
    V = out[1]
    dVdv = out[7]
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


def _gSinT5(v, m, i, e, j, fn):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = v
    vv = vv + vv.T - np.diag(np.diag(vv))
    # Returns: 0=M, 1=V, 2=C, 3=dMdm, 4=dVdm, 5=dCdm, 6=dMdv, 7=dVdv, 8=dCdv
    out = fn(m, vv, i, e, compute_derivatives=True)
    C = out[2]
    dCdv = out[8]
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
