# draw_dp.py
# *Summary:* Draw the double-pendulum system with reward, applied torques, 
# and predictive uncertainty of the tips of the pendulums
#
#    def draw_dp(theta1, theta2, f1, f2, cost, text1=None, text2=None, M=None, S=None)
#
# *Input arguments:*
#
#   theta1     angle of inner pendulum
#   theta2     angle of outer pendulum
#   f1         torque applied to inner pendulum
#   f2         torque applied to outer pendulum
#   cost       cost structure
#     .fcn     function handle (it is assumed to use saturating cost)
#     .<>      other fields that are passed to cost
#   text1      (optional) text field 1
#   text2      (optional) text field 2
#   M          (optional) mean of state
#   S          (optional) covariance of state
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-07

import numpy as np
import matplotlib.pyplot as plt
from pilco_python.scenarios.doublePendulum.getPlotDistr_dp import getPlotDistr_dp
from pilco_python.util.error_ellipse import error_ellipse


def draw_dp(theta1, theta2, f1, f2, cost, text1=None, text2=None, M=None, S=None):
    ## Code
    l = 0.6
    xmin = -2 * l
    xmax = 2 * l
    umax = 2
    height = 0

    # Draw double pendulum
    plt.clf()
    plt.hold(True)
    sth1 = np.sin(theta1)
    sth2 = np.sin(theta2)
    cth1 = np.cos(theta1)
    cth2 = np.cos(theta2)
    pendulum1 = np.array([[0, 0], [-l * sth1, l * cth1]])
    pendulum2 = np.array([[-l * sth1, l * cth1], [-l * (sth1 - sth2), l * (cth1 + cth2)]])
    plt.plot(pendulum1[:, 0], pendulum1[:, 1], 'r', linewidth=4)
    plt.plot(pendulum2[:, 0], pendulum2[:, 1], 'r', linewidth=4)

    # plot target location
    plt.plot(0, 2 * l, 'k+', markersize=20)
    plt.plot([xmin, xmax], [-height, -height], 'k', linewidth=2)
    # plot inner joint
    plt.plot(0, 0, 'k.', markersize=24)
    plt.plot(0, 0, 'y.', markersize=14)
    # plot outer joint
    plt.plot(-l * sth1, l * cth1, 'k.', markersize=24)
    plt.plot(-l * sth1, l * cth1, 'k.', markersize=14)  # original MATLAB uses yellow
    plt.plot(-l * sth1, l * cth1, 'y.', markersize=14)
    # plot tip of outer joint
    plt.plot(-l * (sth1 - sth2), l * (cth1 + cth2), 'k.', markersize=24)
    plt.plot(-l * (sth1 - sth2), l * (cth1 + cth2), 'y.', markersize=14)
    plt.plot(0, -2 * l, '.w', markersize=0.005)

    # % Draw sample positions of the joints
    # if M is not None and S is not None:
    #   samples = gaussian(M, S + 1e-8 * np.eye(4), 1000)
    #   t1 = samples[2, :]
    #   t2 = samples[3, :]
    #   plt.plot(-l * np.sin(t1), l * np.cos(t1), 'b.', markersize=2)
    #   plt.plot(-l * (np.sin(t1) - np.sin(t2)), l * (np.cos(t1) + np.cos(t2)), 'r.', markersize=2)

    # plot ellipses around tips of pendulums (if M, S exist)
    try:
        if M is not None and S is not None and np.max(np.max(S)) > 0:
            M1, S1, M2, S2 = getPlotDistr_dp(M, S, l, l)
            error_ellipse(S1, M1, 0.5, 1, 'style', 'b')  # inner pendulum
            error_ellipse(S2, M2, 0.5, 1, 'style', 'r')  # outer pendulum
    except:
        pass

    # Show other useful information
    # plot applied torques
    plt.plot([0, f1 / umax * xmax], [-0.3, -0.3], 'g', linewidth=10)
    plt.plot([0, f2 / umax * xmax], [-0.5, -0.5], 'g', linewidth=10)
    # plot reward
    # cost expects state [dtheta1, dtheta2, theta1, theta2] -> indices 0,1 are velocity=0
    state = np.array([0, 0, theta1, theta2])
    reward = 1 - cost['fcn'](cost, state, np.zeros((4, 4)))[0]
    plt.plot([0, reward * xmax], [-0.7, -0.7], 'y', linewidth=10)
    plt.text(0, -0.3, 'applied  torque (inner joint)')
    plt.text(0, -0.5, 'applied  torque (outer joint)')
    plt.text(0, -0.7, 'immediate reward')
    if text1 is not None:
        plt.text(0, -0.9, text1)
    if text2 is not None:
        plt.text(0, -1.1, text2)

    plt.gca().set_aspect('equal')
    plt.xlim([xmin, xmax])
    plt.ylim([-2 * l, 2 * l])
    plt.axis('off')
    plt.draw()
