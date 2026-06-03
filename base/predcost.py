# predcost.py
# *Summary:* Compute trajectory of expected costs for a given set of
# state distributions
#
# inputs:
# m0          mean of states, D-by-1 or D-by-K for multiple means
# S           covariance matrix of state distributions
# dynmodel    (dict) for dynamics model (GP)
# plant       (dict) of system parameters
# policy      (dict) for policy to be implemented
# cost        (dict) of cost function parameters
# H           length of optimization horizon
#
# outputs:
# L            expected cumulative (discounted) cost          (H,)
# s            standard deviation of cost                     (H,)
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2012-01-12
#
## High-Level Steps
# # Predict successor state distribution
# # Predict corresponding cost distribution

import numpy as np


def predcost(m0, S, dynmodel, plant, policy, cost, H):
    ## Code

    if m0.ndim == 1:
        m0 = m0.reshape(-1, 1)

    K = m0.shape[1]
    L = np.zeros((K, H))
    s = np.zeros((K, H))

    for k in range(K):
        m = m0[:, k].copy()
        Sk = S.copy()
        for t in range(H):
            m, Sk = plant['prop'](m, Sk, plant, dynmodel, policy)  # get next state
            L_val, d1, d2, v = cost['fcn'](cost, m, Sk)             # compute cost
            L[k, t] = L_val
            s[k, t] = np.sqrt(v)

    L = np.mean(L, axis=0)
    s = np.mean(s, axis=0)
    return L, s
