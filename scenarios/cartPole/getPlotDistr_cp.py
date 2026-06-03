# getPlotDistr_cp.py
# *Summary:* Compute means and covariances of the Cartesian coordinates of
# the tips both the inner and outer pendulum assuming that the joint state
# $x$ of the cart-double-pendulum system is Gaussian, i.e., $x\sim N(m, s)$
#
#     def getPlotDistr_cp(m, s, ell)
#
# *Input arguments:*
#
#   m       mean of full state                                    [4 x 1]
#   s       covariance of full state                              [4 x 4]
#   ell     length of pendulum
#
#   Note: this code assumes that the following order of the state:
#          1: cart pos.,
#          2: cart vel.,
#          3: pendulum angular velocity,
#          4: pendulum angle
#
# *Output arguments:*
#
#   M      mean of tip of pendulum                               [2 x 1]
#   S      covariance of tip of pendulum                         [2 x 2]
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


def getPlotDistr_cp(m, s, ell):
    # Code

    # 1. Augment input distribution to complex angle representation
    m1, s1, c1 = gTrig(m, s, np.array([3]), ell, compute_derivatives=False)
    m1 = np.concatenate([m.ravel(), m1])          # mean of joint
    c1 = s @ c1                                    # cross-covariance between input and prediction
    s1 = np.vstack([np.hstack([s, c1]), np.hstack([c1.T, s1])])  # covariance of joint

    # 2. Compute means of tips of pendulums (in Cartesian coordinates)
    M = np.array([m1[0] + m1[4], -m1[5]])

    # 3. Compute covariances of tips of pendulums (in Cartesian coordinates)
    s11 = s1[0, 0] + s1[4, 4] + s1[0, 4] + s1[4, 0]  # x+l sin(theta)
    s22 = s1[5, 5]  # -l*cos(theta)
    s12 = -(s1[0, 5] + s1[4, 5])  # cov(x+l*sin(th), -l*cos(th)

    S = np.array([[s11, s12], [s12, s22]])

    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        import warnings
        warnings.warn('matrix S not pos.def. (getPlotDistr)')
        eig_min = np.min(np.linalg.eigvalsh(S))
        S = S + (1e-6 - eig_min) * np.eye(2)

    return M, S
