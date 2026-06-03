# lossHinge.py
# *Summary:* Function to compute the moments and derivatives of the loss of a 
# Gaussian distributed point under a double hinge loss function. The loss 
# function has slope -/+a and corners b1 and b2. The function also calculates 
# derivatives of the loss w.r.t. the state distribution.
#
# Graph:
#          \                   /
#           \                 /
#            \               /
#             \_____________/
#             b1           b2
#
# To use a single hinge b1 or b2 can be set to -Inf or +Inf respectively.
#
# Note, this function is only analytic for 1D inputs. To apply this loss
# function to multiple variables, use the lossAdd function.
#
#   def lossHinge(cost, m, s, compute_derivatives=True)
#
# *Input arguments:*
#
#  cost
#    .fcn      lossHinge - called to get here
#    .a        slope of loss function
#    .b        corner points of loss function                     [1   x    2 ]
#  m             input mean                                       [D   x    1 ]
#  S             input covariance matrix                          [D   x    D ]
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
# Last modified: 2013-03-06
#
## High-Level Steps
# # Expected cost
# # Variance of cost
# # inv(s)* cov(x,L)

import numpy as np
from scipy.special import erf


def lossHinge(cost, m, s, compute_derivatives=True):
    ## Code
    D = len(m)

    if D > 1:
        raise ValueError(
            'lossHinge only defined for 1D inputs, use lossAdd to '
            'concatenate multiple 1D loss functions'
        )

    a = cost['a']
    b = cost['b']
    s_val = float(s.ravel()[0])
    m_val = float(np.atleast_2d(m).ravel()[0])

    # centralize
    b_centered = np.atleast_2d(b).ravel() - m_val  # 1x2
    I = ~np.isinf(b_centered)  # valid (non-inf) indices

    eb = np.exp(-b_centered**2 / 2 / s_val)
    erfb = erf(b_centered / np.sqrt(2 * s_val))
    c = np.sqrt(s_val / np.pi / 2)

    # 1. Expected Loss
    # int_{-inf}^{b1-m} -a*(x-b1+m)*N(0,S)  +  int_{b2-m}^inf a*(x-b2+m)*N(0,S)
    L = a * (b_centered / 2 * erfb + c * eb + b_centered * np.array([1, -1]) / 2)
    L = np.sum(L[I])

    dLdm = None
    dLds = None
    S_var = None
    dSdm = None
    dSds = None
    C_out = None
    dCdm = None
    dCds = None

    if compute_derivatives:
        # Derivative w.r.t. m
        dLdb = a / 2 * (erfb + np.array([1, -1]))
        dLdm = np.sum(dLdb[I] * -1) * np.ones((1, 1))

        # Derivative w.r.t. S
        dc = 1 / (2 * np.sqrt(2 * np.pi * s_val))
        dLds = a * np.sum(eb[I]) * dc * np.ones((1, 1))

        # 2. Variance of Loss
        S_var = a**2 * ((b_centered**2 + s_val) * (1 + np.array([1, -1]) * erfb) / 2
                        + np.array([1, -1]) * b_centered * c * eb)
        S_var = np.sum(S_var[I]) - L**2

        erfbdm = -np.sqrt(2 / np.pi / s_val) * eb
        erfbds = -b_centered * eb / np.sqrt(2 * np.pi * s_val**3)
        dSdm_val = a**2 * (-b_centered * (1 + np.array([1, -1]) * erfb)
                           + (b_centered**2 + s_val) * np.array([1, -1]) * erfbdm / 2
                           + np.array([1, -1]) * c * eb * (-1 + b_centered**2 / s_val))
        dSdm_val = np.sum(dSdm_val[I]) - 2 * L * dLdm.item()
        dSdm = np.atleast_2d(dSdm_val)

        dSds_val = a**2 / 2 * ((1 + np.array([1, -1]) * erfb)
                               + (b_centered**2 + s_val) * np.array([1, -1]) * erfbds
                               + np.array([2, -2]) * b_centered * dc * eb
                               + np.array([1, -1]) * b_centered**3 / s_val**2 * c * eb)
        dSds_val = np.sum(dSds_val[I]) - 2 * L * dLds.item()
        dSds = np.atleast_2d(dSds_val)

        # 3. inv(s)* covariance between input and cost
        C_val = a * (np.array([-1, 1]) - erfb) / 2
        C_out = np.atleast_2d(np.sum(C_val)).T

        dCdm = np.atleast_2d(-a / 2 * np.sum(erfbdm[I]))
        dCds = np.atleast_2d(-a / 2 * np.sum(erfbds[I]))

    return L, dLdm, dLds, S_var, dSdm, dSds, C_out, dCdm, dCds
