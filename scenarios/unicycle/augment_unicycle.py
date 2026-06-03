# augment_unicycle.py
# *Summary:* The function computes the $(x,y)$ velocities of the contact point
# in both absolute and unicycle coordinates as well as the the unicycle
# coordinates of the contact point themselves.
#
#       def augment_unicycle(s)
#
# *Input arguments:*
#
#   s     state of the unicycle (including the torques).             [1 x 18]
#         The state is assumed to be given as follows:
#         dx      empty (to be filled by this function)
#         dy      empty (to be filled by this function)
#         dxc     empty (to be filled by this function)
#         dyc     empty (to be filled by this function)
#         dtheta  roll angular velocity
#         dphi    yaw angular velocity
#         dpsiw   wheel angular velocity
#         dpsif   pitch angular velocity
#         dpsit   turn table angular velocity
#         x       x position
#         y       y position
#         xc      empty (to be filled by this function)
#         yc      empty (to be filled by this function)
#         theta   roll angle
#         phi     yaw angle
#         psiw    wheel angle
#         psif    pitch angle
#         psit    turn table angle
#
# *Output arguments:*
#
#   r     additional variables that are computed based on s:          [1 x 6]
#         dx    x velocity of contact point (global coordinates)
#         dy    y velocity of contact point (global coordinates)
#         dxc   x velocity of contact point (unicycle coordinates)
#         dyc   y velocity of contact point (unicycle coordinates)
#         xc    x position of contact point (unicycle coordinates)
#         yc    y position of contact point (unicycle coordinates)
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27

import numpy as np


def augment_unicycle(s):
    ## Code

    rw = 0.225  # wheel radius in meters

    s = np.asarray(s).ravel()

    # state indices (0-based):
    # 0:dx, 1:dy, 2:dxc, 3:dyc, 4:dtheta, 5:dphi, 6:dpsiw, 7:dpsif,
    # 8:dpsit, 9:x, 10:y, 11:xc, 12:yc, 13:theta, 14:phi, 15:psiw,
    # 16:psif, 17:psit

    # x velocity of contact point (global coordinates)
    r0 = rw * np.cos(s[14]) * s[6]
    # y velocity of contact point (global coordinates)
    r1 = rw * np.sin(s[14]) * s[6]

    # (x,y) velocities of contact point (unicycle coordinates)
    A = -np.array([[np.cos(s[14]), np.sin(s[14])],
                   [-np.sin(s[14]), np.cos(s[14])]])
    dA = -s[5] * np.array([[-np.sin(s[14]), np.cos(s[14])],
                            [-np.cos(s[14]), -np.sin(s[14])]])

    r01 = np.array([r0, r1])
    r23 = A @ r01 + dA @ s[9:11]
    # (x,y) coordinates of contact point (unicycle coordinates)
    r45 = A @ s[9:11]

    return np.array([r0, r1, r23[0], r23[1], r45[0], r45[1]])
