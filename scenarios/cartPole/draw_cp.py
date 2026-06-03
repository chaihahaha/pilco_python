# draw_cp.py
# *Summary:* Draw the cart-pole system with reward, applied force, and
# predictive uncertainty of the tip of the pendulum
#
#    def draw_cp(x, theta, force, cost, text1=None, text2=None, M=None, S=None)
#
# *Input arguments:*
#
#   x          position of the cart
#   theta      angle of pendulum
#   force      force applied to cart
#   cost       cost structure (dict)
#     .fcn     function handle (it is assumed to use saturating cost)
#     .<>      other fields that are passed to cost
#   M          (optional) mean of state
#   S          (optional) covariance of state
#   text1      (optional) text field 1
#   text2      (optional) text field 2
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-07

import numpy as np
import matplotlib.pyplot as plt
from .getPlotDistr_cp import getPlotDistr_cp
from pilco_python.util.error_ellipse import error_ellipse


def draw_cp(x, theta, force, cost, text1=None, text2=None, M=None, S=None):
    # Code

    l = 0.6
    xmin = -3
    xmax = 3
    height = 0.1
    width = 0.3
    maxU = 10

    # Compute positions
    cart = np.array([
        [x + width,  height],
        [x + width, -height],
        [x - width, -height],
        [x - width,  height],
        [x + width,  height],
    ])
    pendulum = np.array([
        [x, 0],
        [x + 2 * l * np.sin(theta), -np.cos(theta) * 2 * l],
    ])

    plt.clf()
    plt.hold(True)

    plt.plot(0, 2 * l, 'k+', markersize=20, linewidth=2)
    plt.plot([xmin, xmax], [-height - 0.03, -height - 0.03], 'k', linewidth=2)

    # Plot force
    plt.plot([0, force / maxU * xmax], [-0.3, -0.3], 'g', linewidth=10)

    # Plot reward (without trig augmentation, just evaluate on raw state)
    state_raw = np.array([x, 0, 0, theta])
    reward_val = 1 - cost['fcn'](cost, state_raw, np.zeros((4, 4)))[0]
    if isinstance(reward_val, np.ndarray):
        reward_val = float(reward_val.ravel()[0])
    plt.plot([0, reward_val * xmax], [-0.5, -0.5], 'y', linewidth=10)

    # Plot the cart-pole
    plt.fill(cart[:, 0], cart[:, 1], 'k', edgecolor='k')
    plt.plot(pendulum[:, 0], pendulum[:, 1], 'r', linewidth=4)

    # Plot the joint and the tip
    plt.plot(x, 0, 'y.', markersize=24)
    plt.plot(pendulum[1, 0], pendulum[1, 1], 'y.', markersize=24)

    # plot ellipse around tip of pendulum (if M, S exist)
    if M is not None and S is not None:
        try:
            M1, S1 = getPlotDistr_cp(M, S, 2 * l)
            error_ellipse(S1, M1, style='b')
        except Exception:
            pass

    # Text
    plt.text(0, -0.3, 'applied force')
    plt.text(0, -0.5, 'immediate reward')
    if text1 is not None:
        plt.text(0, -0.9, text1)
    if text2 is not None:
        plt.text(0, -1.1, text2)

    plt.gca().set_aspect('equal')
    plt.xlim(xmin, xmax)
    plt.ylim(-1.4, 1.4)
    plt.axis('off')
    plt.draw()
    plt.show(block=False)
