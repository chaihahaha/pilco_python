# lossSat.py
# *Summary:* Compute expectation and variance of a saturating cost 
# $1 - \exp(-(x-z)^T*W*(x-z)/2)$
# and their derivatives, where x ~ N(m,S), z is a (target state), and W 
# is a weighting matrix
#
#   def lossSat(cost, m, s, compute_derivatives=True)
#
# *Input arguments:*
#
#   cost
#    .z:     target state                                               [D x 1]
#    .W:     weight matrix                                              [D x D]
#  m         mean of input distribution                                 [D x 1]
#  s         covariance matrix of input distribution                    [D x D]
#
# *Output arguments:*
#
#  L               expected loss                                  [1   x    1 ]
#  dLdm            derivative of L wrt input mean                 [1   x    D ]
#  dLds            derivative of L wrt input covariance           [1   x   D^2]
#  S               variance of loss                               [1   x    1 ]
#  dSdm            derivative of S wrt input mean                 [1   x    D ]
#  dSds            derivative of S wrt input covariance           [1   x   D^2]
#  C               inv(S) times input-output covariance           [D   x    1 ]
#  dCdm            derivative of C wrt input mean                 [D   x    D ]
#  dCds            derivative of C wrt input covariance           [D   x   D^2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-28
#
## High-Level Steps
# # Expected cost
# # Variance of cost
# # inv(s)*cov(x,L)

import numpy as np


def lossSat(cost, m, s, compute_derivatives=True):
    ## Code

    # some precomputations
    D = len(m)  # get state dimension

    # set some defaults if necessary
    if 'W' in cost:
        W = cost['W']
    else:
        W = np.eye(D)
    if 'z' in cost:
        z = cost['z']
    else:
        z = np.zeros((D, 1))

    m = np.atleast_2d(m).reshape(-1, 1)
    z = np.atleast_2d(z).reshape(-1, 1)

    SW = s @ W
    # MATLAB: iSpW = W/(eye(D)+SW) solves X*(eye+SW)=W for X
    # => (eye+SW)^T * X^T = W^T => X = solve((eye+SW)^T, W^T)^T
    iSpW = np.linalg.solve((np.eye(D) + SW).T, W.T).T

    # 1. Expected cost, in interval [-1,0]
    L = -np.exp(-(m - z).T @ iSpW @ (m - z) / 2) / np.sqrt(np.linalg.det(np.eye(D) + SW))

    # 1a. derivatives of expected cost
    dLdm = -L * (m - z).T @ iSpW  # wrt input mean
    dLds = L * (iSpW @ (m - z) @ (m - z).T - np.eye(D)) @ iSpW / 2  # wrt input covariance matrix

    S_var = None
    dSdm = None
    dSds = None
    C = None
    dCdm = None
    dCds = None

    if not compute_derivatives:
        return L + 1.0, dLdm, dLds, S_var, dSdm, dSds, C, dCdm, dCds

    # 2. Variance of cost
    # MATLAB: i2SpW = W/(eye(D)+2*SW) => solves X*(eye+2*SW)=W
    i2SpW = np.linalg.solve((np.eye(D) + 2 * SW).T, W.T).T
    r2 = np.exp(-(m - z).T @ i2SpW @ (m - z)) / np.sqrt(np.linalg.det(np.eye(D) + 2 * SW))
    S_var = r2 - L**2
    if S_var < 1e-12:
        S_var = 0.0  # for numerical reasons

    # 2a. derivatives of variance of cost
    # wrt input mean
    dSdm = -2 * r2 * (m - z).T @ i2SpW - 2 * L * dLdm
    # wrt input covariance matrix
    dSds = r2 * (2 * i2SpW @ (m - z) @ (m - z).T - np.eye(D)) @ i2SpW - 2 * L * dLds

    # 3. inv(s)*cov(x,L)
    t = W @ z - iSpW @ (SW @ z + m)
    C = L * t

    dCdm = t @ dLdm - L * iSpW

    # bsxfun(@times,iSpW,permute(t,[3,2,1])) + bsxfun(@times,permute(iSpW,[1,3,2]),t')
    # result[i,j,k] = iSpW[i,j]*t[k] + iSpW[i,k]*t[j]
    t_flat = t.ravel()
    term1 = iSpW[:, :, np.newaxis] * t_flat[np.newaxis, np.newaxis, :]
    term2 = iSpW[:, np.newaxis, :] * t_flat[np.newaxis, :, np.newaxis]
    dc3d = -L * (term1 + term2) / 2
    dc3d_flat = dc3d.reshape(D, D * D, order='F')

    # dCds = bsxfun(@times,t,dLds(:)') + reshape(dCds,D,D^2);
    dCds = t @ dLds.ravel()[np.newaxis, :] + dc3d_flat

    L = 1.0 + L  # bring cost to the interval [0,1]

    return L, dLdm, dLds, S_var, dSdm, dSds, C, dCdm, dCds
