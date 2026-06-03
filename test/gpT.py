# gpT.py
# *Summary:* Test derivatives of gp*-family of functions. It is assumed that
# the gp* function computes the mean and the variance of a GP prediction
# for a Gaussian distributed input x~N(m,s).
# The GP-family of functions is located in <rootDir>/gp and is called gp*.py
#
#
#   (dd, dy, dh) = gpT(deriv, gp, m, s, delta)
#
#
# *Input arguments:*
#
#   deriv    desired derivative. options:
#        (i)    'dMdm' - derivative of the mean of the GP prediction
#                wrt the mean of the input distribution
#        (ii)   'dMds' - derivative of the mean of the GP prediction
#                wrt the variance of the input distribution
#        (iii)  'dMdp' - derivative of the mean of the GP prediction
#                wrt the GP parameters
#        (iv)   'dSdm' - derivative of the variance of the GP prediction
#                wrt the mean of the input distribution
#        (v)    'dSds' - derivative of the variance of the GP prediction
#                wrt the variance of the input distribution
#        (vi)   'dSdp' - derivative of the variance of the GP prediction
#                wrt the GP parameters
#        (vii)  'dVdm' - derivative of inv(s)*(covariance of the input and the
#                GP prediction) wrt the mean of the input distribution
#        (viii) 'dVds' - derivative of inv(s)*(covariance of the input and the
#                GP prediction) wrt the variance of the input distribution
#        (ix)   'dVdp' - derivative of inv(s)*(covariance of the input and the
#                GP prediction) wrt the GP parameters
#   gp       GP structure
#     .fcn   function handle to the GP function used for predictions at
#            uncertain inputs
#     .<>    other fields that are passed on to the GP function
#   m        mean of the input distribution
#   s        covariance of the input distribution
#   delta    (optional) finite difference parameter. Default: 1e-4
#
#
# *Output arguments:*
#
#   dd         relative error of analytical vs. finite difference gradient
#   dy         analytical gradient
#   dh         finite difference gradient
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-06-07

import numpy as np

from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap
from pilco_python.test.checkgrad import checkgrad


def gpT(deriv, gp=None, m=None, s=None, delta=None):
    # set up a default training set and input distribution if not passed in
    if gp is None:
        np.random.seed(123)
        D = 2; E = 1
        gp = {}
        from pilco_python.gp.gp0d import gp0d as gp0d_fn
        gp['fcn'] = gp0d_fn
        # hyp: [D+2 x E] with last row zeros (no noise)
        gp['hyp'] = np.array([
            [0.5],
            [0.3],
            [1.0],
            [0.0],
        ])
        gp['inputs'] = np.array([
            [0.0, 0.0],
            [0.3, 0.1],
            [0.6, 0.4],
            [0.9, 0.7],
            [1.2, 1.1],
            [1.5, 1.5],
            [1.8, 1.9],
            [2.1, 2.3],
            [2.4, 2.7],
            [2.7, 3.0],
        ])
        gp['targets'] = np.sin(gp['inputs'][:, 0]) + 0.5 * gp['inputs'][:, 1]
        gp['targets'] = gp['targets'].reshape(-1, 1)

    if m is None:
        if 'p' in gp and 'inputs' in gp.get('p', {}):
            gp['inputs'] = gp['p']['inputs']
            gp['targets'] = gp['p']['targets']
        D = gp['inputs'].shape[1]
        np.random.seed(456)
        m = np.array([0.5, -0.3])
        r = np.random.randn(D, D)
        s = r @ r.T

    if delta is None:
        delta = 1e-4

    D = len(m)
    gp_fn = gp['fcn']

    # check derivatives
    if deriv == 'dMdm':
        dd, dy, dh = checkgrad(lambda x: _gpT0(x, gp_fn, gp, s), m, delta)
    elif deriv == 'dSdm':
        dd, dy, dh = checkgrad(lambda x: _gpT1(x, gp_fn, gp, s), m, delta)
    elif deriv == 'dVdm':
        dd, dy, dh = checkgrad(lambda x: _gpT2(x, gp_fn, gp, s), m, delta)
    elif deriv == 'dMds':
        idx = np.tril_indices(D)
        dd, dy, dh = checkgrad(lambda x: _gpT3(x, gp_fn, gp, m), s[idx], delta)
    elif deriv == 'dSds':
        idx = np.tril_indices(D)
        dd, dy, dh = checkgrad(lambda x: _gpT4(x, gp_fn, gp, m), s[idx], delta)
    elif deriv == 'dVds':
        idx = np.tril_indices(D)
        dd, dy, dh = checkgrad(lambda x: _gpT5(x, gp_fn, gp, m), s[idx], delta)
    elif deriv == 'dMdp':
        p = unwrap(gp)
        dd, dy, dh = checkgrad(lambda x: _gpT6(x, gp_fn, gp, m, s), p, delta)
    elif deriv == 'dSdp':
        p = unwrap(gp)
        dd, dy, dh = checkgrad(lambda x: _gpT7(x, gp_fn, gp, m, s), p, delta)
    elif deriv == 'dVdp':
        p = unwrap(gp)
        dd, dy, dh = checkgrad(lambda x: _gpT8(x, gp_fn, gp, m, s), p, delta)
    else:
        raise ValueError(f'Unknown deriv: {deriv}')

    return dd, dy, dh


def _gpT0(m, gp_fn, gp, s):
    out = gp_fn(gp, m, s, compute_derivatives=True)
    M = out[0]
    dMdm = out[3]
    f = M
    df = dMdm
    return f, df


def _gpT1(m, gp_fn, gp, s):
    out = gp_fn(gp, m, s, compute_derivatives=True)
    S = out[1]
    dSdm = out[4]
    f = S
    df = dSdm
    return f, df


def _gpT2(m, gp_fn, gp, s):
    out = gp_fn(gp, m, s, compute_derivatives=True)
    V = out[2]
    dVdm = out[5]
    f = V
    df = dVdm
    return f, df


def _gpT3(s_vec, gp_fn, gp, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = gp_fn(gp, m, s_mat, compute_derivatives=True)
    M = out[0]
    dMds = out[6]
    dd_out = len(M)
    dMds = dMds.reshape(dd_out, d, d, order='F')
    df = np.zeros((dd_out, d * (d + 1) // 2))
    for ii in range(dd_out):
        dMdsi = dMds[ii, :, :]
        dMdsi = dMdsi + dMdsi.T - np.diag(np.diag(dMdsi))
        df[ii, :] = dMdsi[tril_idx]
    f = M
    return f, df


def _gpT4(s_vec, gp_fn, gp, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = gp_fn(gp, m, s_mat, compute_derivatives=True)
    M = out[0]
    S = out[1]
    dSds = out[7]
    dd_out = len(M)
    dSds = dSds.reshape(dd_out, dd_out, d, d, order='F')
    df = np.zeros((dd_out, dd_out, d * (d + 1) // 2))
    for ii in range(dd_out):
        for jj in range(dd_out):
            dSdsi = dSds[ii, jj, :, :]
            dSdsi = dSdsi + dSdsi.T - np.diag(np.diag(dSdsi))
            df[ii, jj, :] = dSdsi[tril_idx]
    f = S
    return f, df


def _gpT5(s_vec, gp_fn, gp, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = gp_fn(gp, m, s_mat, compute_derivatives=True)
    M = out[0]
    V = out[2]
    dVds = out[8]
    dd_out = len(M)
    dVds = dVds.reshape(d, dd_out, d, d, order='F')
    df = np.zeros((d, dd_out, d * (d + 1) // 2))
    for ii in range(d):
        for jj in range(dd_out):
            dVdsi = dVds[ii, jj, :, :]
            dVdsi = dVdsi + dVdsi.T - np.diag(np.diag(dVdsi))
            df[ii, jj, :] = dVdsi[tril_idx]
    f = V
    return f, df


def _gpT6(p, gp_fn, gp, m, s):
    gp_copy = dict(gp)
    gp_copy, _ = rewrap(dict(gp), p.copy())
    out = gp_fn(gp_copy, m, s, compute_derivatives=True)
    M = out[0]
    dMdp = out[9]
    f = M
    df = dMdp
    return f, df


def _gpT7(p, gp_fn, gp, m, s):
    gp_copy, _ = rewrap(dict(gp), p.copy())
    out = gp_fn(gp_copy, m, s, compute_derivatives=True)
    S = out[1]
    dSdp = out[10]
    f = S
    df = dSdp
    return f, df


def _gpT8(p, gp_fn, gp, m, s):
    gp_copy, _ = rewrap(dict(gp), p.copy())
    out = gp_fn(gp_copy, m, s, compute_derivatives=True)
    V = out[2]
    dVdp = out[11]
    f = V
    df = dVdp
    return f, df
