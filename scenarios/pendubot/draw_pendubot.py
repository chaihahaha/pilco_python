# draw_pendubot.py
# *Summary:* Draw the Pendubot system with reward, applied torque,
# and predictive uncertainty of the tips of the pendulums
#
#    def draw_pendubot(theta1, theta2, force, cost, text1, text2, M, S)
#
#
# *Input arguments:*
#
#   theta1     angle of inner pendulum
#   theta2     angle of outer pendulum
#   force      torque applied to inner pendulum
#   cost       cost structure (dict)
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
# Last modified: 2013-03-08


def draw_pendubot(theta1, theta2, force, cost, text1=None, text2=None, M=None, S=None):
    ## Code

    import numpy as np
    import matplotlib.pyplot as plt
    from ...util.error_ellipse import error_ellipse
    from .getPlotDistr_pendubot import getPlotDistr_pendubot

    l = 0.6
    xmin = -2 * l
    xmax = 2 * l
    umax = 2.0
    height = 0.0

    # Draw double pendulum
    plt.clf()
    plt.hold(True)

    sth1 = np.sin(theta1)
    sth2 = np.sin(theta2)
    cth1 = np.cos(theta1)
    cth2 = np.cos(theta2)

    pendulum1 = np.array([[0, 0],
                           [-l * sth1, l * cth1]])
    pendulum2 = np.array([[-l * sth1, l * cth1],
                           [-l * (sth1 - sth2), l * (cth1 + cth2)]])

    plt.plot(pendulum1[:, 0], pendulum1[:, 1], 'r', linewidth=4)
    plt.plot(pendulum2[:, 0], pendulum2[:, 1], 'r', linewidth=4)

    # plot target location
    plt.plot(0, 2 * l, 'k+', MarkerSize=20)
    plt.plot([xmin, xmax], [-height, -height], 'k', linewidth=2)
    # plot inner joint
    plt.plot(0, 0, 'k.', markersize=24)
    plt.plot(0, 0, 'y.', markersize=14)
    # plot outer joint
    plt.plot(-l * sth1, l * cth1, 'k.', markersize=24)
    plt.plot(-l * sth1, l * cth1, 'y.', markersize=14)
    # plot tip of outer joint
    plt.plot(-l * (sth1 - sth2), l * (cth1 + cth2), 'k.', markersize=24)
    plt.plot(-l * (sth1 - sth2), l * (cth1 + cth2), 'y.', markersize=14)
    plt.plot(0, -2 * l, '.w', markersize=0.005)

    # # Draw sample positions of the joints
    # if nargin > 6
    #   samples = gaussian(M,S+1e-8*eye(4),1000);
    #   t1 = samples(3,:); t2 = samples(4,:);
    #   plot(-l*sin(t1),l*cos(t1),'b.','markersize',2)
    #   plot(-l*(sin(t1)-sin(t2)),l*(cos(t1)+cos(t2)),'r.','markersize',2)
    # end

    # plot ellipses around tips of pendulums (if M, S exist)
    if M is not None and S is not None:
        try:
            if np.max(np.max(S)) > 0:
                M1, S1, M2, S2 = getPlotDistr_pendubot(M, S, l, l)
                error_ellipse(S1, M1, style='b')  # inner pendulum
                error_ellipse(S2, M2, style='r')  # outer pendulum
        except Exception:
            pass

    # Draw useful information
    # plot applied torque
    plt.plot([0, force / umax * xmax], [-0.5, -0.5], 'g', linewidth=10)
    # plot immediate reward
    reward = 1.0 - cost['fcn'](cost, np.array([0, 0, theta1, theta2]),
                                np.zeros((4, 4)))
    if hasattr(reward, '__iter__'):
        reward = float(np.asarray(reward).ravel()[0])
    plt.plot([0, reward * xmax], [-0.7, -0.7], 'y', linewidth=10)
    plt.text(0, -0.5, 'applied  torque (inner joint)')
    plt.text(0, -0.7, 'immediate reward')
    if text1 is not None:
        plt.text(0, -0.9, text1)
    if text2 is not None:
        plt.text(0, -1.1, text2)

    ax = plt.gca()
    ax.set_aspect(1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-2 * l, 2 * l)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
