# hypCurb.py
# *Summary:* Wrapper for GP training (via gpr.m), penalizing large SNR and
# extreme length-scales to avoid numerical instabilities
#
#     function [f, df] = hypCurb(lh, covfunc, x, y, curb)
#
# *Input arguments:*
#
#   lh       log-hyper-parameters                                   [D+2 x  E ]
#   covfunc  covariance function, e.g.,
#                               covfunc = ('covSum', ('covSEard', 'covNoise'))
#   x        training inputs                                        [ n  x  D ]
#   y        training targets                                       [ n  x  E ]
#   curb     (optional) parameters to penalize extreme hyper-parameters
#     .ls    length-scales
#     .snr   signal-to-noise ratio (try to keep it below 500)
#     .std   additional parameter required for length-scale penalty
#
# *Output arguments:*
#
#   f        penalized negative log-marginal likelihood
#   df       derivative of penalized negative log-marginal likelihood wrt
#            GP log-hyper-parameters
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2011-12-19

import numpy as np
from .gpr import gpr


def hypCurb(lh, covfunc, x, y, curb=None):
    # if nargin < 5, curb.snr = 500; curb.ls = 100; curb.std = 1; end   % set default
    if curb is None:
        curb = {'snr': 500, 'ls': 100, 'std': 1}
    else:
        curb = dict(curb)  # make a copy to avoid modifying caller's dict
    if 'snr' not in curb:
        curb['snr'] = 500
    if 'ls' not in curb:
        curb['ls'] = 100
    if 'std' not in curb:
        curb['std'] = 1

    # p = 30;                                                         % penalty power
    p = 30

    # D = size(x,2);
    D = x.shape[1]

    # if size(lh,1) == 3*D+2; li = 1:2*D; sfi = 2*D+1:3*D+1; % 1D and DD terms
    # elseif size(lh,1) == 2*D+1; li = 1:D; sfi = D+1:2*D;   % Just 1D terms
    # elseif size(lh,1) == D+2; li = 1:D; sfi = D+1;         % Just DD terms
    # else error('Incorrect number of hyperparameters');
    # end
    n_rows = lh.shape[0]
    if n_rows == 3 * D + 2:
        li = slice(0, 2 * D)  # 0:2D (Python 0-indexed = MATLAB 1:2D)
        sfi = slice(2 * D, 3 * D + 1)  # 2D+1:3D+1
    elif n_rows == 2 * D + 1:
        li = slice(0, D)  # 1:D
        sfi = slice(D, 2 * D)  # D+1:2D
    elif n_rows == D + 2:
        li = slice(0, D)  # 1:D
        sfi = D  # D+1 (single index)
    else:
        raise ValueError('Incorrect number of hyperparameters')

    # ll = lh(li); lsf = lh(sfi); lsn = lh(end);
    ll = lh[li] if isinstance(li, slice) else lh[li]  # length-scale log params
    lsf = lh[sfi] if isinstance(sfi, slice) else lh[sfi]  # signal variance log params
    lsn = lh[-1]  # noise log param

    # 1) compute the negative log-marginal likelihood (plus derivatives)
    # [f, df] = gpr(lh, covfunc, x, y);                              % first, call gpr
    f, df = gpr(lh, covfunc, x, y)

    # 2) add penalties and change derivatives accordingly
    # f = f + sum(((ll - log(curb.std'))./log(curb.ls)).^p);   % length-scales
    # Handle curb.std: if it's a scalar, broadcast; if it's an array, use as-is
    curb_std = np.asarray(curb['std']).flatten()
    f = f + np.sum(((ll - np.log(curb_std)) / np.log(curb['ls'])) ** p)

    # df(li) = df(li) + p*(ll - log(curb.std')).^(p-1)/log(curb.ls)^p;
    if isinstance(li, slice):
        df[li] = df[li] + p * (ll - np.log(curb_std)) ** (p - 1) / np.log(curb['ls']) ** p
    else:
        df[li] = df[li] + p * (ll - np.log(curb_std)) ** (p - 1) / np.log(curb['ls']) ** p

    # f = f + sum(((lsf - lsn)/log(curb.snr)).^p); % signal to noise ratio
    f = f + np.sum(((lsf - lsn) / np.log(curb['snr'])) ** p)

    # df(sfi) = df(sfi) + p*(lsf - lsn).^(p-1)/log(curb.snr)^p;
    if isinstance(sfi, slice):
        df[sfi] = df[sfi] + p * (lsf - lsn) ** (p - 1) / np.log(curb['snr']) ** p
    else:
        df[sfi] = df[sfi] + p * (lsf - lsn) ** (p - 1) / np.log(curb['snr']) ** p

    # df(end) = df(end) - p*sum((lsf - lsn).^(p-1)/log(curb.snr)^p);
    df[-1] = df[-1] - p * np.sum((lsf - lsn) ** (p - 1)) / np.log(curb['snr']) ** p

    return f, df
