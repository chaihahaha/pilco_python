# getPlotDistr_cdp.py
# *Summary:* Compute means and covariances of the Cartesian coordinates of
# the tips both the inner and outer pendulum assuming that the joint state
# $x$ of the cart-double-pendulum system is Gaussian, i.e., $x\sim N(m, s)$
#
#
#     def getPlotDistr_cdp(m, s, ell1, ell2)
#
#
# *Input arguments:*
#
#   m       mean of full state                                    [6 x 1]
#   s       covariance of full state                              [6 x 6]
#   ell1    length of inner pendulum
#   ell2    length of outer pendulum
#
#   Note: this code assumes that the following order of the state:
#          0: cart pos.,
#          1: cart vel.,
#          2: pend1 angular velocity,
#          3: pend2 angular velocity,
#          4: pend1 angle,
#          5: pend2 angle
#
# *Output arguments:*
#
#   M1      mean of tip of inner pendulum                         [2 x 1]
#   S1      covariance of tip of inner pendulum                   [2 x 2]
#   M2      mean of tip of outer pendulum                         [2 x 1]
#   S2      covariance of tip of outer pendulum                   [2 x 2]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modification: 2013-03-06
#
## High-Level Steps
# # Augment input distribution to complex angle representation
# # Compute means of tips of pendulums (in Cartesian coordinates)
# # Compute covariances of tips of pendulums (in Cartesian coordinates)
#
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
import warnings
try:
    from ...util.gTrig import gTrig
except (ImportError, ValueError):
    from pilco_python.util.gTrig import gTrig


def getPlotDistr_cdp(m, s, ell1, ell2):
    ## Code

    # 1. Augment input distribution (complex representation)
    # angle indices are [4, 5] (0-indexed), lengths are [ell1, ell2]
    m1, s1, c1 = gTrig(m, s, np.array([4, 5]), np.array([ell1, ell2]),
                        compute_derivatives=False)

    m1_full = np.concatenate([m.ravel(), m1.ravel()])   # mean of joint
    c1_full = s @ c1                                     # cross-covariance between input and prediction
    s1_full = np.block([[s, c1_full],
                        [c1_full.T, s1]])                 # covariance of joint

    # 2. Mean of the tips of the pendulums (Cart. coord.)
    # M1: p2: E[x - l1*sin(theta_2)]; E[l2*cos(theta_2)]
    # In 0-indexed: m1_full[0] = x, m1_full[6] = sin(theta1)*ell1, m1_full[7] = cos(theta1)*ell1
    M1 = np.array([m1_full[0] - m1_full[6], m1_full[7]])
    # p3: mean of cart. coord.
    M2 = np.array([M1[0] - m1_full[8], M1[1] + m1_full[9]])

    # 3. Put covariance matrices together (Cart. coord.)
    # first set of coordinates (tip of 1st pendulum)
    S1 = np.zeros((2, 2))
    S1[0, 0] = s1_full[0, 0] + s1_full[6, 6] - 2*s1_full[0, 6]
    S1[1, 1] = s1_full[7, 7]
    S1[0, 1] = s1_full[0, 7] - s1_full[6, 7]
    S1[1, 0] = S1[0, 1]

    # second set of coordinates (tip of 2nd pendulum)
    S2 = np.zeros((2, 2))
    S2[0, 0] = S1[0, 0] + s1_full[8, 8] + 2*(s1_full[0, 8] - s1_full[6, 8])
    S2[1, 1] = s1_full[7, 7] + s1_full[9, 9] + 2*s1_full[7, 9]
    S2[0, 1] = s1_full[0, 7] - s1_full[6, 7] - s1_full[8, 7] \
               + s1_full[0, 9] - s1_full[6, 9] - s1_full[8, 9]
    S2[1, 0] = S2[0, 1]

    # make sure we have proper covariances (sometimes numerical problems occur)
    try:
        np.linalg.cholesky(S1)
    except np.linalg.LinAlgError:
        warnings.warn('matrix S1 not pos.def. (getPlotDistr)')
        min_eig = np.min(np.linalg.eigvalsh(S1))
        S1 = S1 + (1e-6 - min_eig)*np.eye(2)

    try:
        np.linalg.cholesky(S2)
    except np.linalg.LinAlgError:
        warnings.warn('matrix S2 not pos.def. (getPlotDistr)')
        min_eig = np.min(np.linalg.eigvalsh(S2))
        S2 = S2 + (1e-6 - min_eig)*np.eye(2)

    return M1, S1, M2, S2
