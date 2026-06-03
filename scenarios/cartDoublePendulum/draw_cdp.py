# draw_cdp.py
# *Summary:* Draw the cart-double-pendulum system with reward, applied force,
# and predictive uncertainty of the tips of the pendulums
#
#    def draw_cdp(x, theta2, theta3, force, cost, M=None, S=None, text1=None, text2=None)
#
#
# *Input arguments:*
#
#   x          position of the cart
#   theta2     angle of inner pendulum
#   theta3     angle of outer pendulum
#   force      force applied to cart
#   cost       cost structure
#     'fcn'    function handle (it is assumed to use saturating cost)
#     .<>      other fields that are passed to cost
#   M          (optional) mean of state
#   S          (optional) covariance of state
#   text1      (optional) text field 1
#   text2      (optional) text field 2
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
import matplotlib.pyplot as plt
try:
    from ...util.error_ellipse import error_ellipse
    from .getPlotDistr_cdp import getPlotDistr_cdp
except (ImportError, ValueError):
    from pilco_python.util.error_ellipse import error_ellipse
    from pilco_python.scenarios.cartDoublePendulum.getPlotDistr_cdp import getPlotDistr_cdp


def draw_cdp(x, theta2, theta3, force, cost, M=None, S=None, text1=None, text2=None):
    ## Code

    scale = 1

    l = 0.3*scale
    xmin = -3*scale
    xmax = 3*scale
    height = 0.07*scale
    width  = 0.25*scale

    font_size = 12

    # Compute positions
    cart = np.array([[x + width,  height],
                     [x + width, -height],
                     [x - width, -height],
                     [x - width,  height],
                     [x + width,  height]])

    pend2 = np.array([[x, 0],
                      [x - 2*l*np.sin(theta2), np.cos(theta2)*2*l]])

    pend3 = np.array([[x - 2*l*np.sin(theta2), 2*l*np.cos(theta2)],
                      [x - 2*l*np.sin(theta2) - 2*l*np.sin(theta3),
                       2*l*np.cos(theta2) + 2*l*np.cos(theta3)]])

    # plot cart double pendulum
    plt.clf()
    plt.hold(True)

    plt.plot(0, 4*l, 'k+', markersize=2*font_size, linewidth=2)
    plt.plot([xmin, xmax], [-height-0.03*scale, -height-0.03*scale],
             color='b', linewidth=3)
    plt.plot([0, force/20*xmax], [-0.3, -0.3], color='g', linewidth=font_size)

    # Plot reward
    state = np.array([x, 0, 0, 0, theta2, theta3])
    reward = 1 - cost['fcn'](cost, state, np.zeros((6, 6)))[0]
    plt.plot([0, reward*xmax], [-0.5, -0.5], color='y', linewidth=font_size)

    # Draw Cart
    plt.plot(cart[:, 0], cart[:, 1], color='k', linewidth=3)
    plt.fill(cart[:, 0], cart[:, 1], 'k')
    # Draw Pendulum2
    plt.plot(pend2[:, 0], pend2[:, 1], color='r', linewidth=round(font_size/2))
    # Draw Pendulum3
    plt.plot(pend3[:, 0], pend3[:, 1], color='r', linewidth=round(font_size/2))
    # joint at cart
    plt.plot(x, 0, 'o', markersize=round((font_size+4)/2), color='y',
             markerfacecolor='y')
    # 2nd joint
    plt.plot(pend3[0, 0], pend3[0, 1], 'o', markersize=round((font_size+4)/2),
             color='y', markerfacecolor='y')
    # tip of 2nd joint
    plt.plot(pend3[1, 0], pend3[1, 1], 'o', markersize=round((font_size+4)/2),
             color='y', markerfacecolor='y')

    # plot ellipses around tip of pendulum (if M, S exist)
    try:
        if M is not None and S is not None and np.max(np.abs(S)) > 0:
            M1, S1, M2, S2 = getPlotDistr_cdp(M, S, 2*l, 2*l)
            error_ellipse(S1, M1, style='b')  # inner pendulum
            error_ellipse(S2, M2, style='r')  # outer pendulum
    except Exception:
        pass

    plt.text(0, -0.3*scale, 'applied  force', fontsize=font_size)
    plt.text(0, -0.5*scale, 'immediate reward', fontsize=font_size)

    if text1 is not None:
        plt.text(0, -0.7*scale, text1, fontsize=font_size)
        if text2 is not None:
            plt.text(0, -0.9*scale, text2, fontsize=font_size)

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-1.4, 1.4*scale])

    ax.axis('off')

    plt.draw()
