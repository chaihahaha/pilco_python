# draw_rollout_cdp.py
# *Summary:* Script to draw a trajectory of the cart-double-pendulum system and
#  the predicted uncertainties around the tips of the pendulums
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
## High-Level Steps
# # For each time step, plot the observed trajectory and the predicted
# means and covariances of the Cartesian coordinates of the tips of both
# pendulums
#
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
import time
from .draw_cdp import draw_cdp


def draw_rollout_cdp(xx, latent, M, Sigma, cost, H, dt, j, J, jj):
    """
    Draw a trajectory of the cart-double-pendulum system.

    Parameters
    ----------
    xx : ndarray (N, D_full)
        Full state trajectory data
    latent : list of dict
        List of latent state trajectories
    M : list of ndarray
        Predicted state means
    Sigma : list of ndarray
        Predicted state covariances
    cost : dict
        Cost structure
    H : int
        Prediction horizon
    dt : float
        Time step
    j : int
        Current controlled trial index
    J : int
        Number of initial random trajectories
    jj : int
        Current random trial index
    """
    ## Code

    x_all = xx  # [N, D_full]
    j_val = j   # current trial index

    for r in range(x_all.shape[0]):
        if j_val is not None and M[j_val] is not None and M[j_val].size > 0:
            draw_cdp(latent[j_val][r, 0], latent[j_val][r, 4], latent[j_val][r, 5],
                     latent[j_val][r, -1], cost,
                     M[j_val][:, r], Sigma[j_val][:, :, r],
                     f'trial # {j_val + J}, T={H * dt:.2f} sec',
                     f'total experience (after this trial): {dt * x_all.shape[0]:.2f} sec')
        else:
            draw_cdp(latent[jj][r, 0], latent[jj][r, 4], latent[jj][r, 5],
                     latent[jj][r, -1], cost,
                     None, None,
                     f'(random) trial # {jj}, T={H * dt:.2f} sec',
                     f'total experience (after this trial): {dt * x_all.shape[0]:.2f} sec')
        time.sleep(dt)
