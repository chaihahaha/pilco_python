# draw_rollout_dp.py
# *Summary:* Script to draw a trajectory of the observed double-pendulum system 
# and the predicted uncertainties around the tips of the pendulums
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

import numpy as np
import matplotlib.pyplot as plt
from pilco_python.scenarios.doublePendulum.draw_dp import draw_dp


def draw_rollout_dp(xx, latent, cost, dt, H, M_dict=None, Sigma_dict=None,
                     J=0, j=None, jj=0):
    """
    Loop over states in trajectory and draw.

    Parameters
    ----------
    xx : ndarray
        Observed states (H+1, nX)
    latent : dict or list
        Latent state trajectories for each trial
    cost : dict
        Cost structure
    dt : float
        Sampling time
    H : int
        Horizon length
    M_dict : dict, optional
        Predicted means indexed by trial
    Sigma_dict : dict, optional
        Predicted covariances indexed by trial
    J : int
        Offset for trial numbering
    j : int, optional
        Current trial index (for M/Sigma lookup)
    jj : int
        Trial index for latent lookup
    """
    ## Code

    for r in range(xx.shape[0]):
        cost['t'] = r
        if j is not None and M_dict is not None and M_dict[j] is not None \
           and M_dict[j].size > 0:
            draw_dp(latent[j][r, 2], latent[j][r, 3], latent[j][r, -2],
                    latent[j][r, -1], cost,
                    'trial # %d, T=%g sec' % (j + J, H * dt),
                    'total experience (after this trial): %g sec' % (dt * xx.shape[0]),
                    M_dict[j][:, r], Sigma_dict[j][:, :, r])
        else:
            draw_dp(latent[jj][r, 2], latent[jj][r, 3], latent[jj][r, -2],
                    latent[jj][r, -1], cost,
                    '(random) trial # %d, T=%g sec' % (1, H * dt),
                    'total experience (after this trial): %g sec' % (dt * xx.shape[0]))
        plt.pause(dt)
