# draw_rollout_pendulum.py
# *Summary:* Script to draw a trajectory of the most recent pendulum trajectory
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
## High-Level Steps
# # For each time step, plot the observed trajectory

import numpy as np
import time
from .draw_pendulum import draw_pendulum


def draw_rollout_pendulum(xx, latent, cost, dt, j, J, H, x,
                          M=None, Sigma=None):
    """Loop over states in trajectory and draw each frame.

    Parameters
    ----------
    xx : ndarray
        State trajectory (with controls)
    latent : list of ndarray
        Latent state trajectories per trial
    cost : dict
        Cost structure
    dt : float
        Sampling time
    j : int
        Current trial index (0-based, controlled)
    J : int
        Number of initial random trials
    H : int
        Horizon length
    x : ndarray
        Full training data (state history)
    M : list, optional
        Predicted state means per trial
    Sigma : list, optional
        Predicted state covariances per trial
    """
    ## Code

    # Loop over states in trajectory
    for r in range(xx.shape[0]):
        cost['t'] = r
        if j is not None and M is not None and M[j] is not None and M[j].size > 0:
            draw_pendulum(
                latent[j][r, 1],
                latent[j][r, -1],
                cost,
                'trial # %d, T=%d sec' % (j + J + 1, H * dt),
                'total experience (after this trial): %d sec' % (dt * x.shape[0]),
                M[j][:, r],
                Sigma[j][:, :, r]
            )
        else:
            draw_pendulum(
                latent[0][r, 1],
                latent[0][r, -1],
                cost,
                '(random) trial # %d, T=%d sec' % (1, H * dt),
                'total experience (after this trial): %d sec' % (dt * x.shape[0])
            )
        time.sleep(dt)
