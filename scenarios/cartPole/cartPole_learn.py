# cartPole_learn.py
# *Summary:* Script to learn a controller for the cart-pole swingup
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
## High-Level Steps
# # Load parameters
# # Create J initial trajectories by applying random controls
# # Controlled learning (train dynamics model, policy learning, policy
# application)

import numpy as np
import os
import sys

# Add project root for package imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pilco_python.scenarios.cartPole.settings_cp import define_settings
from pilco_python.scenarios.cartPole.loss_cp import loss_cp
from pilco_python.base.rollout import rollout
from pilco_python.base.learnPolicy import learn_policy
from pilco_python.base.applyController import applyController
from pilco_python.util.gaussian import gaussian


def run_cartPole_learn(basename='cartPole_', random_seed=1):
    # 1. Initialization
    np.random.seed(random_seed)

    settings = define_settings()
    mu0 = settings['mu0']
    S0 = settings['S0']
    plant = settings['plant']
    policy = settings['policy']
    cost = settings['cost']
    H = settings['H']
    J = settings['J']
    N = settings['N']
    dt = settings['dt']
    plotting = settings['plotting']
    dyno = settings['dyno']
    odei = settings['odei']
    poli = settings['poli']

    x = settings['x']
    y = settings['y']
    fantasy_mean = settings['fantasy_mean']
    fantasy_std = settings['fantasy_std']
    realCost = settings['realCost']
    M = settings['M']
    Sigma = settings['Sigma']

    mu0Sim = settings['mu0Sim']
    S0Sim = settings['S0Sim']

    latent = {}

    # 2. Initial J random rollouts
    for jj in range(J):
        start_state = gaussian(mu0, S0)
        xx, yy, real_cost_traj, latent_traj = rollout(start_state, policy, H, plant, cost)
        realCost[jj] = real_cost_traj
        latent[jj] = latent_traj

        x = np.vstack([x, xx])
        y = np.vstack([y, yy])

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            from .draw_rollout_cp import draw_rollout_cp
            print(f'(random) trial # {jj + 1}, T={H * dt:.1f} sec')
            draw_rollout_cp(xx, latent, M, Sigma, jj, -1, H, dt, x, cost)

    # 3. Controlled learning (N iterations)
    for j in range(N):
        print(f'--- controlled trial # {j + 1} ---')

        # trainDynModel placeholder
        # train(dynmodel, x, y)  -- would require GP training infrastructure

        # learnPolicy
        policy, fX3, M_j, Sigma_j, fantasy_mean_j, fantasy_std_j = \
            learn_policy(mu0Sim, S0Sim, settings['dynmodel'], policy, plant,
                         cost, H, plotting)

        fantasy_mean[j] = fantasy_mean_j
        fantasy_std[j] = fantasy_std_j
        M[j] = M_j
        Sigma[j] = Sigma_j

        # applyController
        x, y, realCost, latent = applyController(
            plant, cost, policy, mu0, S0, H, H * 2, j, J,
            x, y, basename, M, Sigma, dyno,
            plots_verbosity=plotting['verbosity'],
            realCost=realCost, latent=latent,
            rollout_fn=rollout,
        )

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            from .draw_rollout_cp import draw_rollout_cp
            print(f'controlled trial # {j + 1}, T={H * dt:.1f} sec')
            draw_rollout_cp(x, latent, M, Sigma, j, J, H, dt, x, cost, j=j)

    return policy, cost, plant, settings


if __name__ == '__main__':
    run_cartPole_learn()
