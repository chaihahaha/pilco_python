# draw_rollout_cp.py
# *Summary:* Script to draw the most recent trajectory of the cart-pole
# system together with the predicted uncertainties around the tip of the
# pendulum
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-20
#
## High-Level Steps
# # For each time step, plot the observed trajectory and the predicted
# means and covariances of the Cartesian coordinates of the tip of the
# pendulum

import numpy as np
import matplotlib.pyplot as plt
from .draw_cp import draw_cp


def draw_rollout_cp(xx, latent, M, Sigma, jj, J, H, dt, x, cost, j=None):
    # Code
    # xx: observed states [H x (nX + 2*nA + nU)]
    # latent: latent states {trial: [H+1 x nX+nU]}
    # M: predicted means {trial: [D x H+1]}
    # Sigma: predicted covariances {trial: [D x D x H+1]}

    if j is not None:
        trial_idx = j + J
        M_j = M[j]
        Sigma_j = Sigma[j]
    else:
        trial_idx = jj

    # Loop over states in trajectory (= time steps)
    for r in range(len(xx)):
        if j is not None and M_j is not None and M_j.size > 0:
            draw_cp(latent[j][r, 0], latent[j][r, 3], latent[j][r, -1], cost,
                    f'trial # {trial_idx + 1}, T={H * dt:.1f} sec',
                    f'total experience (after this trial): {dt * len(x):.1f} sec',
                    M_j[:, r], Sigma_j[:, :, r])
        else:
            draw_cp(latent[jj][r, 0], latent[jj][r, 3], latent[jj][r, -1], cost,
                    f'(random) trial # {jj + 1}, T={H * dt:.1f} sec',
                    f'total experience (after this trial): {dt * len(x):.1f} sec')

        plt.pause(dt)
