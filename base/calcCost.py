# calcCost.py
# *Summary:* Function to calculate the incurred cost and its standard deviation,
# given a sequence of predicted state distributions and the cost dict
#
#    L, sL = calcCost(cost, M, S)
#
# *Input arguments:*
#
#   cost               cost structure (dict)
#   M                  mean vectors of state trajectory (D-by-H matrix)
#   S                  covariance matrices at each time step (D-by-D-by-H)
#
# *Output arguments:*
#
#   L                  expected incurred cost of state trajectory  (H,)
#   sL                 standard deviation of incurred cost         (H,)
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-01-23
#
## High-Level Steps
# # Augment state distribution with trigonometric functions
# # Compute distribution of the control signal
# # Compute dynamics-GP prediction
# # Compute distribution of the next state
#

import numpy as np

def calcCost(cost, M, S):
    ## Code

    # ensure S is 3D (D x D x H)
    if S.ndim == 2:
        S = S[:, :, np.newaxis]

    H = M.shape[1]                                          # horizon length
    L = np.zeros(H)
    SL = np.zeros(H)

    # for each time step, compute the expected cost and its variance
    for h in range(H):
        L_val, d1, d2, SL_val = cost['fcn'](cost, M[:, h], S[:, :, h])
        L[h] = L_val.item()
        SL[h] = SL_val.item()

    sL = np.sqrt(SL)                                        # standard deviation
    return L, sL
