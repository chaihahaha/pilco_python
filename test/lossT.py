# lossT.py
# *Summary:* Test derivatives of cost functions. It is assumed that
# the cost function computes (at least) the mean and the variance of the
# cost for a Gaussian distributed input $x\sim\mathcal N(m,s)$
#
#
#   function [dd dy dh] = lossT(deriv, policy, m, s, delta)
#
#
# *Input arguments:*
#
#   deriv    desired derivative. options:
#         (i)   'dMdm' - derivative of the mean of the predicted cost
#                wrt the mean of the input distribution
#         (ii)  'dMds' - derivative of the mean of the predicted cost
#                wrt the variance of the input distribution
#         (iii) 'dSdm' - derivative of the variance of the predicted cost
#                wrt the mean of the input distribution
#         (iv)  'dSds' - derivative of the variance of the predicted cost
#                wrt the variance of the input distribution
#         (v)   'dCdm' - derivative of inv(s)*(covariance of the input and the
#                predicted cost) wrt the mean of the input distribution
#         (vi)  'dCds' - derivative of inv(s)*(covariance of the input and the
#                predicted cost) wrt the variance of the input distribution
#   cost     cost structure
#     .fcn   function handle to cost
#     .<>    other fields that are passed on to the cost
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
from pilco_python.loss.lossQuad import lossQuad


def _losstest01(m, cost, S):
    """dLdm"""
    res = cost['fcn'](cost, m, S)
    L, dLdm = res[0], res[1]
    f = L
    df = dLdm
    return f, df


def _losstest02(s, cost, m):
    """dLds"""
    d = len(m)
    ss = np.zeros((d, d))
    ss[np.tril_indices(d)] = s
    ss = ss + ss.T - np.diag(np.diag(ss))

    res = cost['fcn'](cost, m, ss)
    L, dLdm, dLds = res[0], res[1], res[2]

    f = L
    df = dLds
    df = 2 * df - np.diag(np.diag(df))
    df = df[np.tril_indices(d)]
    return f, df


def _losstest03(m, cost, S):
    """dSdm"""
    res = cost['fcn'](cost, m, S)
    L, dLdm, dLds, S_val, dSdm = res[0], res[1], res[2], res[3], res[4]

    f = S_val
    df = dSdm
    return f, df


def _losstest04(s, cost, m):
    """dSds"""
    d = len(m)
    ss = np.zeros((d, d))
    ss[np.tril_indices(d)] = s
    ss = ss + ss.T - np.diag(np.diag(ss))

    res = cost['fcn'](cost, m, ss)
    L, dLdm, dLds, S_val, dSdm, dSds = res[0], res[1], res[2], res[3], res[4], res[5]

    f = S_val
    df = dSds
    df = 2 * df - np.diag(np.diag(df))
    df = df[np.tril_indices(d)]
    return f, df


def _losstest05(m, cost, S):
    """dCdm"""
    res = cost['fcn'](cost, m, S)
    L, dLdm, dLds, S_val, dSdm, dSds, C, dCdm = \
        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]

    f = C
    df = dCdm
    return f, df


def _losstest06(s, cost, m):
    """dCds"""
    d = len(m)
    ss = np.zeros((d, d))
    ss[np.tril_indices(d)] = s
    ss = ss + ss.T - np.diag(np.diag(ss))

    res = cost['fcn'](cost, m, ss)
    L, dLdm, dLds, S_val, dSdm, dSds, C, dCdm, dCds = \
        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]

    f = C
    dCds = np.reshape(dCds, (d, d, d))
    df = np.zeros((d, d * (d + 1) // 2))
    for i in range(d):
        dCdsi = np.squeeze(dCds[i, :, :])
        dCdsi = dCdsi + dCdsi.T - np.diag(np.diag(dCdsi))
        df[i, :] = dCdsi[np.tril_indices(d)]
    return f, df


def lossT(deriv=None, cost=None, m=None, S=None, delta=None):
    ## Code

    # create a default test if no input arguments are given
    if deriv is None:
        D = 4
        # deterministic values instead of randn
        m = np.array([[0.5], [-0.2], [1.3], [-0.7]])
        np.random.seed(42)
        Stmp = np.random.randn(D, D)
        S = Stmp @ Stmp.T
        cost = {}
        cost['z'] = np.array([[0.1], [0.8], [-0.5], [0.3]])
        np.random.seed(43)
        Wtmp = np.random.randn(D, D)
        W = Wtmp @ Wtmp.T
        cost['W'] = W
        cost['fcn'] = lossQuad
        deriv = 'dLdm'

    D = len(m)
    if delta is None:
        delta = 1e-4

    # check derivatives
    if deriv in ('dLdm', 'dMdm'):
        d, dy, dh = checkgrad(_losstest01, m, delta, cost, S)

    elif deriv in ('dLds', 'dMds'):
        d, dy, dh = checkgrad(_losstest02, S[np.tril_indices(D)], delta, cost, m)

    elif deriv == 'dSdm':
        d, dy, dh = checkgrad(_losstest03, m, delta, cost, S)

    elif deriv == 'dSds':
        d, dy, dh = checkgrad(_losstest04, S[np.tril_indices(D)], delta, cost, m)

    elif deriv in ('dCdm', 'dVdm'):
        d, dy, dh = checkgrad(_losstest05, m, delta, cost, S)

    elif deriv in ('dCds', 'dVds'):
        d, dy, dh = checkgrad(_losstest06, S[np.tril_indices(D)], delta, cost, m)

    return d, dy, dh
