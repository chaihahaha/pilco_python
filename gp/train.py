# train.py
# *Summary:* Train a GP model with SE covariance function (ARD). First, the
# hyper-parameters are trained using a full GP. Then, if gpmodel.induce exists,
# indicating sparse approximation, if enough training examples are present,
# train the inducing inputs (hyper-parameters are taken from the full GP).
# If no inducing inputs are present, then initialize them to be a
# random subset of the training inputs.
#
#   function [gpmodel, nlml] = train(gpmodel, dump, iter)
#
# *Input arguments:*
#
#   gpmodel                 GP structure
#     inputs                GP training inputs                      [ N  x  D]
#     targets               GP training targets                     [ N  x  E]
#     hyp (optional)        GP log-hyper-parameters                 [D+2 x  E]
#     induce (optional)     pseudo inputs for sparse GP
#   dump                    not needed for this code, but required
#                           for compatibility reasons
#   iter                    optimization iterations for training    [1   x  2]
#                           [full GP, sparse GP]
#
# *Output arguments:*
#
#   gpmodel                 updated GP structure
#   nlml                    negative log-marginal likelihood
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2016-07-11

import numpy as np
from ..util.minimize import minimize
from .hypCurb import hypCurb
from .fitc import fitc


def train(gpmodel, dump=None, iter=None):
    # 1) Initialization
    # if nargin < 3, iter = [-500 -1000]; end           % default training iterations
    if iter is None:
        iter = np.array([-500, -1000])

    # D = size(gpmodel.inputs,2); E = size(gpmodel.targets,2);   % get variable sizes
    D = gpmodel['inputs'].shape[1]
    E = gpmodel['targets'].shape[1]

    # covfunc = {'covSum', {'covSEard', 'covNoise'}};        % specify ARD covariance
    covfunc = ('covSum', ('covSEard', 'covNoise'))

    # curb.snr = 1000; curb.ls = 100; curb.std = std(gpmodel.inputs);   % set hyp curb
    curb = {
        'snr': 1000,
        'ls': 100,
        'std': np.std(gpmodel['inputs'], axis=0, ddof=1)
    }

    # if ~isfield(gpmodel,'hyp')  % if we don't pass in hyper-parameters, define them
    if 'hyp' not in gpmodel:
        gpmodel['hyp'] = np.zeros((D + 2, E))
        nlml = np.zeros(E)
        # lh = repmat([log(std(gpmodel.inputs)) 0 -1]',1,E);   % init hyp length scales
        # lh(D+1,:) = log(std(gpmodel.targets));                      %  signal std dev
        # lh(D+2,:) = log(std(gpmodel.targets)/10);                     % noise std dev
        # MATLAB: repmat([log(std(inputs)) 0 -1]', 1, E)
        # This creates (D+2, E) where each column is [log(std(inputs)), 0, -1]
        init_row = np.concatenate([np.log(np.std(gpmodel['inputs'], axis=0, ddof=1)), [0], [-1]])
        lh = np.tile(init_row.reshape(-1, 1), (1, E))
        lh[D, :] = np.log(np.std(gpmodel['targets'], axis=0, ddof=1))  # signal std dev (index D = D+1 in MATLAB)
        lh[D + 1, :] = np.log(np.std(gpmodel['targets'], axis=0, ddof=1) / 10)  # noise std dev (index D+1 = D+2 in MATLAB)
    else:
        lh = gpmodel['hyp'].copy()                                       # GP hyper-parameters
        nlml = np.zeros(E)

    # 2a) Train full GP (always)
    print('Train hyper-parameters of full GP ...')
    for i in range(E):                                          # train each GP separately
        print(f'GP {i + 1}/{E}')
        # try   % BFGS training
        try:
            optimized_hyp, v, _, _ = minimize(
                lh[:, i].copy(), hypCurb, int(iter[0]), covfunc,
                gpmodel['inputs'], gpmodel['targets'][:, i], curb
            )
        except Exception:
            # catch % conjugate gradients (BFGS can be quite aggressive)
            optimized_hyp, v, _, _ = minimize(
                lh[:, i].copy(), hypCurb,
                {'length': int(iter[0]), 'method': 'CG', 'verbosity': 1},
                covfunc, gpmodel['inputs'], gpmodel['targets'][:, i], curb
            )
        gpmodel['hyp'][:, i] = optimized_hyp.flatten()
        nlml[i] = v[-1]

    # 2b) If necessary: sparse training using FITC/SPGP (Snelson & Ghahramani, 2006)
    # if isfield(gpmodel,'induce')            % are we using a sparse approximation?
    if 'induce' in gpmodel:
        # [N D] = size(gpmodel.inputs);
        N = gpmodel['inputs'].shape[0]
        # [M uD uE] = size(gpmodel.induce);
        if gpmodel['induce'].ndim == 3:
            M, uD, uE = gpmodel['induce'].shape
        else:
            M = gpmodel['induce'].shape[0]
            uD = gpmodel['induce'].shape[1]
            uE = 1
            gpmodel['induce'] = gpmodel['induce'].reshape(M, uD, uE)

        # if M >= N; return; end     % if too few training examples, we don't need FITC
        if M >= N:
            return gpmodel, nlml

        print('Train pseudo-inputs of sparse GP ...')

        # if uD == 0                               % we don't have inducing inputs yet?
        if uD == 0:
            # gpmodel.induce = zeros(M,D,uE);                            % allocate space
            gpmodel['induce'] = np.zeros((M, D, uE))
            # iter = 3*iter; % train a lot for the first time (it's still cheap!)
            iter = 3 * iter
            # for i = 1:uE
            #    j = randperm(N);
            #    gpmodel.induce(:,:,i) = gpmodel.inputs(j(1:M),:);       % random subset
            # end
            uE = gpmodel['induce'].shape[2]  # update after reshape
            for e_idx in range(uE):
                j = np.random.permutation(N)
                gpmodel['induce'][:, :, e_idx] = gpmodel['inputs'][j[:M], :]

        # train sparse model
        # [gpmodel.induce nlml2] = minimize(gpmodel.induce, 'fitc', iter(end), gpmodel);
        # minimize expects a callable; fitc takes (induce, gpmodel)
        optimized_induce, nlml2, _, _ = minimize(
            gpmodel['induce'].copy(), fitc, int(iter[-1]), gpmodel
        )
        gpmodel['induce'] = optimized_induce

        print(f'GP NLML, full: {np.sum(nlml):e}, sparse: {nlml2[-1]:e}, diff: {nlml2[-1] - np.sum(nlml):e}')

    return gpmodel, nlml
