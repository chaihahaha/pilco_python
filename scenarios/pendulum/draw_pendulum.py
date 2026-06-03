# draw_pendulum.py
# *Summary:* Draw the pendulum system with reward, applied torque,
# and predictive uncertainty of the tips of the pendulums
#
#    def draw_pendulum(theta, torque, cost, text1, text2, M, S)
#
#
# *Input arguments:*
#
#   theta     angle of pendulum
#   torque    torque applied to pendulum
#   cost      cost dict
#     .fcn    function handle (it is assumed to use saturating cost)
#     .<>     other fields that are passed to cost
#   text1     (optional) text field 1
#   text2     (optional) text field 2
#   M         (optional) mean of state
#   S         (optional) covariance of state
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-18

import numpy as np
import matplotlib.pyplot as plt


def draw_pendulum(theta, torque, cost, text1=None, text2=None, M=None, S=None):
    ## Code

    l = 0.6
    xmin = -1.2 * l
    xmax = 1.2 * l
    umax = 0.5
    height = 0.0

    plt.clf()

    # Draw pendulum
    pendulum_x = [0.0, l * np.sin(theta)]
    pendulum_y = [0.0, -l * np.cos(theta)]
    plt.plot(pendulum_x, pendulum_y, 'r', linewidth=4)

    # plot ellipses around tips of pendulum (if M, S exist)
    if M is not None and S is not None:
        if np.max(np.abs(S)) > 0:
            err = np.linspace(-1, 1, 100) * np.sqrt(S[1, 1])
            plt.plot(l * np.sin(M[1] + 2 * err), -l * np.cos(M[1] + 2 * err), 'b', linewidth=1)
            plt.plot(l * np.sin(M[1] + err), -l * np.cos(M[1] + err), 'b', linewidth=2)
            plt.plot(l * np.sin(M[1]), -l * np.cos(M[1]), 'b.', markersize=20)

    # Draw useful information
    # target location
    plt.plot(0, l, 'k+', markersize=20)
    plt.plot([xmin, xmax], [-height, -height], 'k', linewidth=2)
    # joint
    plt.plot(0, 0, 'k.', markersize=24)
    plt.plot(0, 0, 'y.', markersize=14)
    # tip of pendulum
    plt.plot(l * np.sin(theta), -l * np.cos(theta), 'k.', markersize=24)
    plt.plot(l * np.sin(theta), -l * np.cos(theta), 'y.', markersize=14)
    plt.plot(0, -2 * l, '.w', markersize=0.005)
    # applied torque
    plt.plot([0, torque / umax * xmax], [-0.5, -0.5], 'g', linewidth=10)
    # immediate reward
    reward = 1.0 - cost['fcn'](cost, np.array([[0.0], [theta]]), np.zeros((2, 2)))[0]
    plt.plot([0, reward * xmax], [-0.7, -0.7], 'y', linewidth=10)
    plt.text(0, -0.5, 'applied  torque')
    plt.text(0, -0.7, 'immediate reward')
    if text1 is not None:
        plt.text(0, -0.9, text1)
    if text2 is not None:
        plt.text(0, -1.1, text2)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-2 * l, 2 * l])
    ax.axis('off')
    plt.draw()
