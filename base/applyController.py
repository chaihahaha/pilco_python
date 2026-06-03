# applyController.py
# *Summary:* Apply the learned controller to a (simulated) system
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-06-04
#
## High-Level Steps
# # Generate a single trajectory rollout by applying the controller
# # Generate many rollouts for testing the performance of the controller
# # Save the data

import numpy as np
import matplotlib.pyplot as plt


# from ..util.gaussian import gaussian
# from .rollout import rollout


def _gaussian(m, S, n=1):
    """Local copy to avoid import issues during testing."""
    D = S.shape[0]
    L = np.linalg.cholesky(S)
    Z = np.random.randn(D, n)
    x = m.reshape(-1, 1) + L @ Z
    return x


def applyController(plant, cost, policy, mu0, S0, H, maxH, j, J,
                    x, y, basename, M, Sigma, dyno,
                    plots_verbosity=0,
                    realCost=None, latent=None,
                    rollout_fn=None):
    ## Code

    if realCost is None:
        realCost = {}
    if latent is None:
        latent = {}

    # 1. Generate trajectory rollout given the current policy
    if 'constraint' in plant:
        HH = maxH
    else:
        HH = H

    start_state = _gaussian(mu0, S0)

    if rollout_fn is None:
        raise ValueError("rollout_fn must be provided for applyController")

    xx, yy, real_cost_traj, latent_traj = rollout_fn(start_state, policy, HH, plant, cost)

    realCost[j + J] = real_cost_traj
    latent[j] = latent_traj

    print(xx)                                  # display states of observed trajectory

    x = np.vstack([x, xx])
    y = np.vstack([y, yy])                     # augment training set

    if plots_verbosity > 0:
        plt.figure(3)
        # plt.hold(True)  # deprecated in matplotlib 2.0+
        plt.plot(range(1, len(realCost[j + J]) + 1), realCost[j + J], 'r')
        plt.draw()

    # 2. Make many rollouts to test the controller quality
    if plots_verbosity > 1:
        lat = [None] * 10
        for i in range(10):
            _, _, _, lat[i] = rollout_fn(start_state, policy, HH, plant, cost)

        plt.figure(4)
        plt.clf()

        ldyno = len(dyno)
        for i in range(ldyno):       # plot the rollouts on top of predicted error bars
            plt.subplot(int(np.ceil(ldyno / np.sqrt(ldyno))),
                        int(np.ceil(np.sqrt(ldyno))), i + 1)
            # plt.hold(True)  # deprecated in matplotlib 2.0+

            # predicted error bars
            pred_len = M[j].shape[1]
            sigma_vals = 2 * np.sqrt(Sigma[j][i, i, :])
            plt.errorbar(range(pred_len), M[j][i, :], yerr=sigma_vals)

            for ii in range(10):
                lat_seq = lat[ii][:, dyno[i]]
                plt.plot(range(len(lat_seq)), lat_seq, 'r')

            latent_seq = latent[j][:, dyno[i]]
            plt.plot(range(len(latent_seq)), latent_seq, 'g')
            plt.axis('tight')

        plt.draw()

    # 3. Save data
    filename = basename + str(j) + '_H' + str(H)
    np.savez(filename, x=x, y=y, realCost=realCost, latent=latent)

    return x, y, realCost, latent
