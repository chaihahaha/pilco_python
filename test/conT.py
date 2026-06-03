# conT.py
# *Summary:* Test derivatives of controller functions. It is assumed that
# the controller function computes the mean and the variance of the
# control signal for a Gaussian distributed input x~N(m,s)
#
#
#   (dd, dy, dh) = conT(deriv, policy, m, s, delta)
#
#
# *Input arguments:*
#
#   deriv    desired derivative. options:
#        (i)    'dMdm'
#        (ii)   'dMds'
#        (iii)  'dMdp'
#        (iv)   'dSdm'
#        (v)    'dSds'
#        (vi)   'dSdp'
#        (vii)  'dCdm'
#        (viii) 'dCds'
#        (ix)   'dCdp'
#   policy   policy structure
#     .fcn   callable controller function
#     .<>    other fields that are passed on to the controller
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
# Last modified: 2013-05-30

import numpy as np

from pilco_python.test.checkgrad import checkgrad


def conT(deriv, policy=None, m=None, s=None, delta=None):
    if delta is None:
        delta = 1e-4

    # if no input arguments, create random policy parameters
    if policy is None:
        np.random.seed(789)
        D_ctrl = 2   # number of controllers (E in conCat terms)
        d_state = 3  # state dimension (D in conCat terms)
        w = np.array([
            [0.5, -0.3, 0.2],
            [-0.1, 0.8, -0.4],
        ])
        b = np.array([0.1, -0.2])
        m = np.array([1.0, -0.5, 0.3])
        r = np.random.randn(d_state, d_state)
        s = r @ r.T
        s = (s + s.T) / 2
        maxU = np.array([2.0, 3.0])

        from pilco_python.control.conCat import conCat
        from pilco_python.control.conLin import conlin
        from pilco_python.util.gSat import gSat

        policy = {
            'p': {'w': w, 'b': b},
            'maxU': maxU,
            'fcn': lambda policy, m, s, compute_d=True:
                conCat(conlin, gSat, policy, m, s, compute_derivatives=compute_d)
        }

    D_ctrl = len(policy['maxU'])
    d_state = len(m)

    if deriv == 'dMdm':
        dd, dy, dh = checkgrad(lambda x: _conT0(x, policy, s), m, delta)
    elif deriv == 'dSdm':
        dd, dy, dh = checkgrad(lambda x: _conT1(x, policy, s), m, delta)
    elif deriv == 'dCdm':
        dd, dy, dh = checkgrad(lambda x: _conT2(x, policy, s), m, delta)
    elif deriv == 'dMds':
        idx = np.tril_indices(d_state)
        dd, dy, dh = checkgrad(lambda x: _conT3(x, policy, m), s[idx], delta)
    elif deriv == 'dSds':
        idx = np.tril_indices(d_state)
        dd, dy, dh = checkgrad(lambda x: _conT4(x, policy, m), s[idx], delta)
    elif deriv == 'dCds':
        idx = np.tril_indices(d_state)
        dd, dy, dh = checkgrad(lambda x: _conT5(x, policy, m), s[idx], delta)
    elif deriv == 'dMdp':
        from pilco_python.util.unwrap import unwrap
        p = unwrap(policy['p'])
        dd, dy, dh = checkgrad(lambda x: _conT6(x, policy, m, s), p, delta)
    elif deriv == 'dSdp':
        from pilco_python.util.unwrap import unwrap
        p = unwrap(policy['p'])
        dd, dy, dh = checkgrad(lambda x: _conT7(x, policy, m, s), p, delta)
    elif deriv == 'dCdp':
        from pilco_python.util.unwrap import unwrap
        p = unwrap(policy['p'])
        dd, dy, dh = checkgrad(lambda x: _conT8(x, policy, m, s), p, delta)
    else:
        raise ValueError(f'Unknown deriv: {deriv}')

    return dd, dy, dh


def _conT0(m, policy, s):
    out = policy['fcn'](policy, m, s, compute_d=True)
    M = out[0]
    dMdm = out[3]
    f = M
    df = dMdm
    return f, df


def _conT1(m, policy, s):
    out = policy['fcn'](policy, m, s, compute_d=True)
    S = out[1]
    dSdm = out[4]
    f = S
    df = dSdm
    return f, df


def _conT2(m, policy, s):
    out = policy['fcn'](policy, m, s, compute_d=True)
    C = out[2]
    dCdm = out[5]
    f = C
    df = dCdm
    return f, df


def _conT3(s_vec, policy, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = policy['fcn'](policy, m, s_mat, compute_d=True)
    M = out[0]
    dMds = out[6]
    dd_out = len(M)
    dMds = dMds.reshape(dd_out, d, d, order='F')
    df_out = np.zeros((dd_out, d * (d + 1) // 2))
    for ii in range(dd_out):
        dMds_i = dMds[ii, :, :]
        dMds_i = dMds_i + dMds_i.T - np.diag(np.diag(dMds_i))
        df_out[ii, :] = dMds_i[tril_idx]
    f = M
    df = df_out
    return f, df


def _conT4(s_vec, policy, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = policy['fcn'](policy, m, s_mat, compute_d=True)
    S = out[1]
    dSds = out[7]
    dd_out = S.shape[0]
    dSds = dSds.reshape(dd_out, dd_out, d, d, order='F')
    df_out = np.zeros((dd_out, dd_out, d * (d + 1) // 2))
    for ii in range(dd_out):
        for jj in range(dd_out):
            dSds_ij = dSds[ii, jj, :, :]
            dSds_ij = dSds_ij + dSds_ij.T - np.diag(np.diag(dSds_ij))
            df_out[ii, jj, :] = dSds_ij[tril_idx]
    f = S
    df = df_out
    return f, df


def _conT5(s_vec, policy, m):
    d = len(m)
    tril_idx = np.tril_indices(d)
    vv = np.zeros((d, d))
    vv[tril_idx] = s_vec
    s_mat = vv + vv.T - np.diag(np.diag(vv))
    out = policy['fcn'](policy, m, s_mat, compute_d=True)
    C = out[2]
    dCds = out[8]
    dd_out = C.shape[0]
    dCds = dCds.reshape(d, dd_out, d, d, order='F')
    df_out = np.zeros((d, dd_out, d * (d + 1) // 2))
    for ii in range(d):
        for jj in range(dd_out):
            dCds_ij = dCds[ii, jj, :, :]
            dCds_ij = dCds_ij + dCds_ij.T - np.diag(np.diag(dCds_ij))
            df_out[ii, jj, :] = dCds_ij[tril_idx]
    f = C
    df = df_out
    return f, df


def _conT6(p, policy, m, s):
    from pilco_python.util.rewrap import rewrap
    policy_copy = dict(policy)
    policy_copy['p'], _ = rewrap(policy['p'].copy(), p.copy())
    out = policy['fcn'](policy_copy, m, s, compute_d=True)
    M = out[0]
    dMdp = out[9]
    f = M
    df = dMdp
    return f, df


def _conT7(p, policy, m, s):
    from pilco_python.util.rewrap import rewrap
    policy_copy = dict(policy)
    policy_copy['p'], _ = rewrap(policy['p'].copy(), p.copy())
    out = policy['fcn'](policy_copy, m, s, compute_d=True)
    S = out[1]
    dSdp = out[10]
    f = S
    df = dSdp
    return f, df


def _conT8(p, policy, m, s):
    from pilco_python.util.rewrap import rewrap
    policy_copy = dict(policy)
    policy_copy['p'], _ = rewrap(policy['p'].copy(), p.copy())
    out = policy['fcn'](policy_copy, m, s, compute_d=True)
    C = out[2]
    dCdp = out[11]
    f = C
    df = dCdp
    return f, df
