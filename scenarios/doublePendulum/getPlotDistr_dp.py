# getPlotDistr_dp.py
# *Summary:* Compute means and covariances of the Cartesian coordinates of
# the tips both the inner and outer pendulum assuming that the joint state
# $x$ of the cart-double-pendulum system is Gaussian, i.e., $x\sim N(m, s)$
#
#
#     def getPlotDistr_dp(m, s, ell1, ell2)
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
#          1: pend1 angular velocity,
#          2: pend2 angular velocity, 
#          3: pend1 angle, 
#          4: pend2 angle
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
# Last modification: 2013-03-27
#
## High-Level Steps
# # Augment input distribution to complex angle representation
# # Compute means of tips of pendulums (in Cartesian coordinates)
# # Compute covariances of tips of pendulums (in Cartesian coordinates)

import numpy as np
from pilco_python.util.gTrig import gTrig


def getPlotDistr_dp(m, s, ell1, ell2):
    ## Code

    # 1. Augment input distribution
    # map input through sin/cos, angle indices [3,4] in MATLAB 1-based -> [2,3] in Python 0-based
    m1, s1, c1 = gTrig(m, s, np.array([2, 3]), np.array([ell1, ell2]), compute_derivatives=False)
    m1 = np.concatenate([m, m1])         # mean of joint
    c1 = s @ c1                           # cross-covariance between input and prediction
    s1 = np.block([[s, c1], [c1.T, s1]])  # covariance of joint

    # 2. Compute means of tips of pendulums (in Cartesian coordinates)
    M1 = np.array([-m1[4], m1[5]])                   # [-l*sin(t1), l*cos(t1)]
    M2 = np.array([-m1[4] + m1[6], m1[5] + m1[7]])   # [-l*(sin(t1)-sin(t2)),l*(cos(t1)+cos(t2))]

    # 2. Put covariance matrices together (Cart. coord.)
    # first set of coordinates (tip of 1st pendulum)
    s11 = s1[4, 4]
    s12 = -s1[4, 5]
    s22 = s1[5, 5]
    S1 = np.array([[s11, s12], [s12.T, s22]])

    # second set of coordinates (tip of 2nd pendulum)
    s11 = s1[4, 4] + s1[6, 6] - s1[4, 6] - s1[6, 4]    # ell1*sin(t1) + ell2*sin(t2)
    s22 = s1[5, 5] + s1[7, 7] + s1[5, 7] + s1[7, 5]    # ell1*cos(t1) + ell2*cos(t2)
    s12 = -(s1[4, 5] + s1[4, 7] + s1[6, 5] + s1[6, 7])
    S2 = np.array([[s11, s12], [s12.T, s22]])

    # make sure we have proper covariances (sometimes numerical problems occur)
    try:
        np.linalg.cholesky(S1)
    except np.linalg.LinAlgError:
        print('matrix S1 not pos.def. (getPlotDistr)')
        eig_min = np.min(np.linalg.eigh(S1)[0])
        S1 = S1 + (1e-6 - eig_min) * np.eye(2)

    try:
        np.linalg.cholesky(S2)
    except np.linalg.LinAlgError:
        print('matrix S2 not pos.def. (getPlotDistr)')
        eig_min = np.min(np.linalg.eigh(S2)[0])
        S2 = S2 + (1e-6 - eig_min) * np.eye(2)

    return M1, S1, M2, S2
