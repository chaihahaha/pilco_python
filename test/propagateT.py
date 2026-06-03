# propagateT.py
# *Summary:* Test derivatives of the propagate function, which computes the
# mean and the variance of the successor state distribution, assuming that the
# current state is Gaussian distributed with mean m and covariance matrix
# s.
#
#   [dd dy dh] = propagateT(deriv, plant, dynmodel, policy, m, s, delta)
#
#
# *Input arguments:*
#
#   deriv    desired derivative. options:
#        (i)    'dMdm' - derivative of the mean of the predicted state
#                wrt the mean of the input distribution
#        (ii)   'dMds' - derivative of the mean of the predicted state
#                wrt the variance of the input distribution
#        (iii)  'dMdp' - derivative of the mean of the predicted state
#                wrt the controller parameters
#        (iv)   'dSdm' - derivative of the variance of the predicted state
#                wrt the mean of the input distribution
#        (v)    'dSds' - derivative of the variance of the predicted state
#                wrt the variance of the input distribution
#        (vi)   'dSdp' - derivative of the variance of the predicted state
#                wrt the controller parameters
#        (vii)  'dCdm' - derivative of the inv(s)*(covariance of the input and
#                the predicted state) wrt the mean of the input distribution
#        (viii) 'dCds' - derivative of the inv(s)*(covariance of the input and
#                the predicted state) wrt the variance of the input distribution
#        (ix)   'dCdp' - derivative of the inv(s)*(covariance of the input and
#                the predicted state) wrt the controller parameters
#   plant      plant structure
#     .poli    state indices: policy inputs
#     .dyno    state indices: predicted variables
#     .dyni    state indices: inputs to ODE solver
#     .difi    state indices that are learned via differences
#     .angi    state indices: angles
#     .poli    state indices: policy inputs
#     .prop    function handle to function responsible for state
#              propagation. Here: @propagated
#   dynmodel   GP dynamics model (structure)
#     .hyp     log-hyper parameters
#     .inputs  training inputs
#     .targets training targets
#   policy     policy structure
#     .maxU    maximum amplitude of control
#     .fcn     function handle to policy
#     .p       struct of policy parameters
#     .p.<>    policy-specific parameters are stored here
#   m          mean of the input distribution
#   s          covariance of the input distribution
#   delta      (optional) finite difference parameter. Default: 1e-4
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
# Last modified: 2013-03-21

import numpy as np

from pilco_python.test.checkgrad import checkgrad
from pilco_python.base.propagated import propagated
from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def _simple_linear_sat_policy(policy, m, S, compute_derivatives=False):
    """
    Simple linear + saturating policy stub for default test setup.
    Replaces MATLAB's conCat(@conlin, @gSat, policy, m, s).
    Once conCat/conlin are ported, use them instead.
    """
    import numpy as np

    m = np.asarray(m).ravel()
    S = np.atleast_2d(S)
    W = np.asarray(policy['p']['w'])
    b = np.asarray(policy['p']['b']).ravel()
    maxU = np.asarray(policy['maxU'])
    F = len(b)
    D = len(m)

    M = W @ m + b
    V = W @ S @ W.T
    V = (V + V.T) / 2
    C = W.T

    if not compute_derivatives:
        return M, V, C

    dMdm = W.copy()
    dVdm = np.zeros((F * F, D))
    dCdm = np.zeros((D * F, D))
    dMdv = np.zeros((F, D * D))
    dVdv = np.kron(W, W)
    dCdv = np.zeros((D * F, D * D))

    P = W.size + b.size
    Mdp = np.zeros((F, P))
    Sdp = np.zeros((F * F, P))
    Cdp = np.zeros((D * F, P))

    return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv, Mdp, Sdp, Cdp


def _propagateT0(m, plant, dynmodel, policy, s):
    """dMdm"""
    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, s, plant, dynmodel, policy, compute_derivatives=True)
    df = dMdm
    f = M
    return f, df


def _propagateT1(m, plant, dynmodel, policy, s):
    """dSdm"""
    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, s, plant, dynmodel, policy, compute_derivatives=True)
    dd = len(M)
    df = np.reshape(dSdm, (dd, dd, -1))
    f = S
    return f, df


def _propagateT2(s_vec, plant, dynmodel, policy, m):
    """dMds"""
    d = len(m)
    ss = np.zeros((d, d))
    ss[np.tril_indices(d)] = s_vec
    ss = ss + ss.T - np.diag(np.diag(ss))

    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, ss, plant, dynmodel, policy, compute_derivatives=True)

    dd = len(M)
    dMds_r = np.reshape(dMds, (dd, d, d))
    df = np.zeros((dd, d * (d + 1) // 2))
    for i in range(dd):
        dMdsi = dMds_r[i, :, :]
        dMdsi = dMdsi + dMdsi.T - np.diag(np.diag(dMdsi))
        df[i, :] = dMdsi[np.tril_indices(d)]
    f = M
    return f, df


def _propagateT3(s_vec, plant, dynmodel, policy, m):
    """dSds"""
    d = len(m)
    ss = np.zeros((d, d))
    ss[np.tril_indices(d)] = s_vec
    ss = ss + ss.T - np.diag(np.diag(ss))

    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, ss, plant, dynmodel, policy, compute_derivatives=True)

    dd = len(M)
    dSds_r = np.reshape(dSds, (dd, dd, d, d))
    df = np.zeros((dd, dd, d * (d + 1) // 2))
    for i in range(dd):
        for j in range(dd):
            dSdsi = np.squeeze(dSds_r[i, j, :, :])
            dSdsi = dSdsi + dSdsi.T - np.diag(np.diag(dSdsi))
            df[i, j, :] = dSdsi[np.tril_indices(d)]
    f = S
    return f, df


def _propagateT4(p_vec, plant, dynmodel, policy, m, s):
    """dMdp"""
    policy['p'] = rewrap(policy['p'], p_vec)[0]

    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, s, plant, dynmodel, policy, compute_derivatives=True)

    df = dMdp
    f = M
    return f, df


def _propagateT5(p_vec, plant, dynmodel, policy, m, s):
    """dSdp"""
    policy['p'] = rewrap(policy['p'], p_vec)[0]

    M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp = \
        propagated(m, s, plant, dynmodel, policy, compute_derivatives=True)

    df = dSdp
    f = S
    return f, df


def propagateT(deriv=None, plant=None, dynmodel=None, policy=None, m=None, s=None, delta=None):
    ## Code

    if deriv is None or plant is None:
        E = 4
        F = 3
        D = E

        # deterministic values instead of randn('seed',24)
        m = np.array([[0.3], [-0.8], [1.2], [-0.4]])
        np.random.seed(24)
        S_tmp = np.random.randn(D, D)
        s = S_tmp @ S_tmp.T

        # Plant ----------------------------------------------------------------
        plant = {}
        plant['poli'] = np.arange(E)
        plant['dyno'] = np.arange(E)
        plant['dyni'] = np.arange(E)
        plant['difi'] = np.arange(E)
        plant['angi'] = np.array([], dtype=int)
        plant['prop'] = propagated

        # Policy ---------------------------------------------------------------
        policy = {}
        np.random.seed(25)
        policy['p'] = {}
        policy['p']['w'] = np.random.randn(F, E)
        np.random.seed(26)
        policy['p']['b'] = np.random.randn(F, 1)
        policy['fcn'] = _simple_linear_sat_policy
        policy['maxU'] = 20 * np.ones(F)

        # Dynamics -------------------------------------------------------------
        nn = 10
        dynmodel = {}
        dynmodel['hyp'] = np.zeros((1, E))
        np.random.seed(27)
        dynmodel['inputs'] = np.random.randn(nn, E + F)
        np.random.seed(28)
        dynmodel['targets'] = np.random.randn(nn, E)

        if deriv is None:
            deriv = 'dMdm'

    if delta is None:
        delta = 1e-4

    D = len(m)

    if deriv == 'dMdm':
        dd, dy, dh = checkgrad(_propagateT0, m, delta, plant, dynmodel, policy, s)

    elif deriv == 'dSdm':
        dd, dy, dh = checkgrad(_propagateT1, m, delta, plant, dynmodel, policy, s)

    elif deriv == 'dMds':
        dd, dy, dh = checkgrad(_propagateT2, s[np.tril_indices(D)], delta, plant,
                               dynmodel, policy, m)

    elif deriv == 'dSds':
        dd, dy, dh = checkgrad(_propagateT3, s[np.tril_indices(D)], delta, plant,
                               dynmodel, policy, m)

    elif deriv == 'dMdp':
        p = unwrap(policy['p'])
        dd, dy, dh = checkgrad(_propagateT4, p, delta, plant, dynmodel, policy, m, s)

    elif deriv == 'dSdp':
        p = unwrap(policy['p'])
        dd, dy, dh = checkgrad(_propagateT5, p, delta, plant, dynmodel, policy, m, s)

    return dd, dy, dh
