# unicycle_learn.py
# *Summary:* Script to learn a controller for unicycling
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
# High-Level Steps
# # Load parameters
# # Create J initial trajectories by applying random controls
# # Controlled learning (train dynamics model, policy learning, policy application)

import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pilco_python.util.gaussian import gaussian
from pilco_python.util.gTrig import gTrig
from pilco_python.base.rollout import rollout
from pilco_python.base.trainDynModel import train_dyn_model
from pilco_python.base.learnPolicy import learn_policy
from pilco_python.base.applyController import apply_controller

from pilco_python.scenarios.unicycle.settings_unicycle import *
from pilco_python.scenarios.unicycle.dynamics_unicycle import dynamics_unicycle
from pilco_python.scenarios.unicycle.augment_unicycle import augment_unicycle
from pilco_python.scenarios.unicycle.loss_unicycle import loss_unicycle
from pilco_python.scenarios.unicycle.draw_rollout_unicycle import draw_rollout_unicycle

from pilco_python.gp.gp1d import gp1d
from pilco_python.gp.train import train
from pilco_python.control.conCat import conCat
from pilco_python.control.conlin import conlin
from pilco_python.util.gSat import gSat
from pilco_python.base.propagated import propagated

# Wire up function references
plant['dynamics'] = dynamics_unicycle
plant['augment'] = augment_unicycle
plant['propagate'] = propagated

policy['fcn'] = lambda p, m, s: conCat(conlin, gSat, p, m, s)

cost['fcn'] = loss_unicycle

dynmodel['fcn'] = gp1d
dynmodel['train'] = train

basename = 'unicycle_'

# set random seeds (deterministic for reproducibility)
np.random.seed(1)


def main():
    ## Code

    global x, y, realCost, M_cell, Sigma_cell, fantasy_mean, fantasy_std
    global plotting, policy, plant, cost, dynmodel, opt, trainOpt, basename

    odei = plant['odei']
    augi = plant['augi']
    dyno = plant['dyno']
    dyni = plant['dyni']

    # 1. Initialization
    print('Starting unicycle PILCO learning...')

    # 2. Initial J random rollouts
    latent_list = [None] * J
    for jj in range(J):
        start_state = gaussian(mu0, S0, 1).ravel()
        xx, yy, L_cost, latent = rollout(
            start_state,
            {'maxU': policy['maxU'] / 5.0},
            H, plant, cost, compute_cost=True)
        x = xx if x is None else np.vstack([x, xx])
        y = yy if y is None else np.vstack([y, yy])
        realCost[jj] = L_cost
        latent_list[jj] = latent

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.clf()
            draw_rollout_unicycle(plt.gcf(), xx, plant, dt, cost,
                                   j=1, J_val=J, H=H, x=x, jj=jj)

    print('Completed %d random rollouts' % J)

    # Compute distribution of augmented start state via MCMC
    z_odei = mu0.reshape(-1, 1) + np.linalg.cholesky(S0).T @ np.random.randn(len(odei), 1000)
    z_aug = np.zeros((len(odei) + len(augi), 1000))
    z_aug[np.sort(np.concatenate([odei, augi])), :] = 0  # placeholder
    for i_idx in range(z_odei.shape[1]):
        aug_vals = plant['augment'](z_odei[:, i_idx])
        z_aug[:len(odei), i_idx] = z_odei[:, i_idx]
        z_aug[len(odei):, i_idx] = aug_vals
    mu0Sim = np.mean(z_aug, axis=1)
    S0Sim = np.cov(z_aug)
    mu0Sim[odei] = mu0
    S0Sim[np.ix_(odei, odei)] = S0
    mu0Sim = mu0Sim[dyno]
    S0Sim = S0Sim[np.ix_(dyno, dyno)]

    # 3. Controlled learning (N iterations)
    for j in range(N):
        print('Controlled trial #%d' % (j + 1))

        train_dyn_model(x, y, dynmodel, policy, plant, trainOpt)

        learn_policy(mu0Sim, S0Sim, dynmodel, policy, plant, cost, H)

        xx, yy, L_cost, latent_new = apply_controller(
            plant, policy, cost, mu0, S0, H, j)

        x = np.vstack([x, xx]) if x is not None else xx
        y = np.vstack([y, yy]) if y is not None else yy
        realCost[j + J] = L_cost
        if j + J < len(latent_list):
            latent_list[j + J] = latent_new
        else:
            latent_list.append(latent_new)

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.clf()
            draw_rollout_unicycle(plt.gcf(), xx, plant, dt, cost,
                                   j=j + 1, J_val=J, H=H, x=x)

    print('Learning completed.')


if __name__ == '__main__':
    main()
