# draw_rollout_pendubot
# *Summary:* Script to draw the most recent trajectory of the Pendubot system
# and the predicted uncertainties around the tips of the pendulums
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
#
## High-Level Steps
# # For each time step, plot the observed trajectory and the predicted
# means and covariances of the Cartesian coordinates of the tips of both
# pendulums

import numpy as np
from .draw_pendubot import draw_pendubot


def draw_rollout_pendubot(j, J, xx, latent, M, Sigma, cost, H, dt, x):
    ## Code

    # Loop over states in trajectory
    for r in range(xx.shape[0]):
        cost['t'] = r
        if j is not None and M[j] is not None and M[j].size > 0:
            draw_pendubot(
                latent[j][r, 2], latent[j][r, 3],
                latent[j][r, -1], cost,
                'trial # %d, T=%.2f sec' % (j + J, H * dt),
                'total experience (after this trial): %.2f sec' %
                (dt * x.shape[0]),
                M[j][:, r] if M[j].ndim > 1 and M[j].shape[1] > r else None,
                Sigma[j][:, :, r] if Sigma[j].ndim > 2 and Sigma[j].shape[2] > r else None
            )
        else:
            draw_pendubot(
                xx[r, 2], xx[r, 3],
                xx[r, -1] if xx.shape[1] > 4 else 0.0,
                cost,
                '(random) trial # %d, T=%.2f sec' % (0, H * dt),
                'total experience (after this trial): %.2f sec' %
                (dt * x.shape[0])
            )
