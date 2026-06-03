# pred.py
# *Summary:* Compute predictive (marginal) distributions of a trajectory
#
#   M, S = pred(policy, plant, dynmodel, m, s, H)
#
# *Input arguments:*
#
#   policy             policy structure (dict)
#   plant              plant structure (dict)
#   dynmodel           dynamics model structure (dict)
#   m                  D-by-1 mean of the initial state distribution
#   s                  D-by-D covariance of the initial state distribution
#   H                  length of prediction horizon
#
# *Output arguments:*
#
#   M                  D-by-(H+1) sequence of predicted mean vectors
#   S                  D-by-D-by-(H+1) sequence of predicted covariance
#                      matrices
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-01-23
#
## High-Level Steps
# # Predict successor state distribution

import numpy as np


def pred(policy, plant, dynmodel, m, s, H):
    ## Code

    D = len(m)
    S = np.zeros((D, D, H + 1))
    M = np.zeros((D, H + 1))
    M[:, 0] = m.flatten()
    S[:, :, 0] = s

    for i in range(H):
        m, s = plant['prop'](m, s, plant, dynmodel, policy)
        M[:, i + 1] = m[-D:].flatten()
        S[:, :, i + 1] = s[-D:, -D:]

    return M, S
