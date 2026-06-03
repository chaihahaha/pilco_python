# reward.py
# *Summary:* Compute expectation, variance, and their derivatives of an 
# exponentiated negative quadratic cost $\exp(-(x-z)'W(x-z)/2)$,
# where $x\sim\mathcal N(m,S)$
#
#   def reward(m, S, z, W, compute_derivatives=True)
#
# *Input arguments:*
#
#  m:          D-by-1 mean of the state distribution
#  S:          D-by-D covariance matrix of the state distribution
#  z:          D-by-D weight matrix
#  W:          D-by-D weight matrix
#
# *Output arguments:*
#
#  muR:        1-by-1 expected reward
#  dmuRdm:     1-by-D derivative of expected reward wrt input mean
#  dmuRdS:     D-by-D derivative of expected reward wrt input covariance matrix
#  sR:         1-by-1 variance of reward
#  dsRdm:      1-by-D derivative of variance of reward wrt input mean
#  dsRdS:      D-by-D derivative reward variance wrt input covariance matrix
#
# Copyright (C) 2008-2013 by 
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen. 
#
# Last modification: 2013-01-20
#
## High-Level Steps
# # Compute expected reward
# # Compute the derivatives of the expected reward with respect to the input 
#   distribution (optional)
# # Compute variance of reward
# # Compute the derivatives of the variance of the reward with
# respect to the input distribution (optional)

import numpy as np


def reward(m, S, z, W, compute_derivatives=True):
    ## Code

    # some precomputations
    D = len(m)  # get state dimension

    m = np.atleast_2d(m).reshape(-1, 1)
    z = np.atleast_2d(z).reshape(-1, 1)

    SW = S @ W
    # MATLAB: W/(eye(D)+SW) solves X*(I+SW)=W for X, i.e., X = W*inv(I+SW)
    # Equivalently: solve((I+SW)^T, W^T)^T
    iSpW = np.linalg.solve((np.eye(D) + SW).T, W.T).T

    # 1. expected reward
    muR = np.exp(-(m - z).T @ iSpW @ (m - z) / 2) / np.sqrt(np.linalg.det(np.eye(D) + SW))

    dmuRdm = None
    dmuRdS = None
    sR = None
    dsRdm = None
    dsRdS = None

    if not compute_derivatives:
        return muR, dmuRdm, dmuRdS, sR, dsRdm, dsRdS

    # 2. derivatives of expected reward
    dmuRdm = -muR * (m - z).T @ iSpW  # wrt input mean
    dmuRdS = muR * (iSpW @ (m - z) @ (m - z).T - np.eye(D)) @ iSpW / 2  # wrt input covariance matrix

    # 3. reward variance
    i2SpW = np.linalg.solve((np.eye(D) + 2 * SW).T, W.T).T
    r2 = np.exp(-(m - z).T @ i2SpW @ (m - z)) / np.sqrt(np.linalg.det(np.eye(D) + 2 * SW))
    sR = r2 - muR**2
    if sR < 1e-12:
        sR = 0.0  # for numerical reasons

    # 4. derivatives of reward variance
    # wrt input mean
    dsRdm = -2 * r2 * (m - z).T @ i2SpW - 2 * muR * dmuRdm
    # wrt input covariance matrix
    dsRdS = r2 * (2 * i2SpW @ (m - z) @ (m - z).T - np.eye(D)) @ i2SpW - 2 * muR * dmuRdS

    return muR, dmuRdm, dmuRdS, sR, dsRdm, dsRdS
