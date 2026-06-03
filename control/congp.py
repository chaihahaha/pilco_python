# congp.py
# *Summary:* Implements the mean-of-GP policy (equivalent to a regularized RBF
# network. Compute mean, variance and input-output covariance of
# the control $u$ using a mean-of-GP policy function, when the input $x$ is
# Gaussian. The GP is parameterized using a pseudo training set size N.
# Optionally, compute partial derivatives wrt the input parameters.
#
# This version sets the signal variance to 1, the noise to 0.01 and their
# respective lengthscales to zero. This results in only the lengthscales,
# inputs, and outputs being trained.
#
#   function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
#            = congp(policy, m, s)
#
# *Input arguments:*
#
#   policy        policy (dict)
#     'p'         parameters that are modified during training
#       'hyp'     GP-log hyperparameters (Ph = (d+2)*D)              [ Ph      ]
#       'inputs'  policy pseudo inputs                               [ N  x  d ]
#       'targets' policy pseudo targets                              [ N  x  D ]
#   m             mean of state distribution                         [ d  x  1 ]
#   s             covariance matrix of state distribution            [ d  x  d ]
#
# *Output arguments:*
#
#   M             mean of the predicted control                      [ D  x  1 ]
#   S             covariance of predicted control                    [ D  x  D ]
#   C             inv(s)*covariance between input and control        [ d  x  D ]
#   dMdm          deriv. of mean control wrt mean of state           [ D  x  d ]
#   dSdm          deriv. of control variance wrt mean of state       [D*D x  d ]
#   dCdm          deriv. of covariance wrt mean of state             [d*D x  d ]
#   dMds          deriv. of mean control wrt variance                [ D  x d*d]
#   dSds          deriv. of control variance wrt variance            [D*D x d*d]
#   dCds          deriv. of covariance wrt variance                  [d*D x d*d]
#   dMdp          deriv. of mean control wrt GP hyper-parameters     [ D  x  P ]
#   dSdp          deriv. of control variance wrt GP hyper-parameters [D*D x  P ]
#   dCdp          deriv. of covariance wrt GP hyper-parameters       [d*D x  P ]
#
# where P = (d+2)*D + n*(d+D) is the total number of policy parameters.
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-01-24
#
# High-Level Steps
# # Extract policy parameters from policy structure
# # Compute predicted control u inv(s)*covariance between input and control
# # Set derivatives of non-free parameters to zero
# # Merge derivatives
#
# Translated from MATLAB 1-indexed to Python 0-indexed.

import numpy as np
from types import SimpleNamespace
from ..gp.gp2 import gp2
from ..gp.gp2d import gp2d


def congp(policy, m, s, compute_derivatives=False):
    # 1. Extract policy parameters
    policy_hyp = policy['p']['hyp'].copy()
    policy_inputs = policy['p']['inputs']
    policy_targets = policy['p']['targets']

    # fix policy signal and the noise variance
    # (avoids some potential numerical problems)
    policy_hyp[-2] = np.log(1)          # set signal variance to 1
    policy_hyp[-1] = np.log(0.01)       # set noise standard dev to 0.01

    # Build the gp model object for gp2/gp2d (they use attribute access)
    # Reshape hyp from flat vector to [d+2, E] matrix
    d_in = policy_inputs.shape[1]
    E_out = policy_targets.shape[1]
    hyp_mat = policy_hyp.reshape(d_in + 2, E_out, order='F')
    gp_model = SimpleNamespace(
        hyp=hyp_mat,
        inputs=policy_inputs,
        targets=policy_targets,
    )

    # 2. Compute predicted control u inv(s)*covariance between input and control
    if not compute_derivatives:
        M, S, C = gp2(gp_model, m, s)
        return M, S, C

    else:
        (M, S, C,
         dMdm, dSdm, dCdm,
         dMds, dSds, dCds,
         dMdi, dSdi, dCdi,
         dMdt, dSdt, dCdt,
         dMdX, dSdX, dCdX) = gp2d(gp_model, m, s)

        # 3. Set derivatives of non-free parameters to zero: signal and noise variance
        d = policy_inputs.shape[1]                  # input dimension
        dimU = policy_targets.shape[1]               # output / control dimension
        d2 = d + 2                                   # number of hyperparams per output

        # In MATLAB: sidx = bsxfun(@plus, (d+1:d2)', (0:dimU-1)*d2)
        # The rows (d+1:d2) = last two hyperparams (signal var, noise) in 1-indexed
        # In 0-indexed: indices d, d+1 for each output block
        row_idx = np.arange(d, d2)                   # [d, d+1]  (signal var, noise rows)
        col_offsets = np.arange(dimU) * d2            # [0, d2, 2*d2, ..., (dimU-1)*d2]
        sidx = (row_idx[:, None] + col_offsets).ravel(order='F')

        dMdX[:, sidx] = 0
        dSdX[:, sidx] = 0
        dCdX[:, sidx] = 0

        # 4. Merge derivatives
        dMdp = np.hstack([dMdX, dMdi, dMdt])
        dSdp = np.hstack([dSdX, dSdi, dSdt])
        dCdp = np.hstack([dCdX, dCdi, dCdt])

        return (M, S, C,
                dMdm, dSdm, dCdm,
                dMds, dSds, dCds,
                dMdp, dSdp, dCdp)
