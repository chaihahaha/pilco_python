# pendulum_learn.py
# *Summary:* Script to learn a controller for the pendulum swingup
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
#
## High-Level Steps
# # Load parameters
# # Create J initial trajectories by applying random controls
# # Controlled learning (train dynamics model, policy learning, policy application)

import numpy as np
import matplotlib.pyplot as plt


def pendulum_learn(settings=None):
    """Run the PILCO pendulum swingup learning loop.

    Parameters
    ----------
    settings : dict or None
        If None, load from settings_pendulum.get_settings().
        Otherwise, use the provided settings dict.

    Returns
    -------
    settings : dict
        Updated settings after learning (contains x, y, policy, dynmodel, etc.)
    """
    # Imports are inside to avoid circular imports at module level
    from .settings_pendulum import get_settings
    from .dynamics_pendulum import dynamics_pendulum
    from .loss_pendulum import loss_pendulum
from ...control.conCat import conCat
from ...control.congp import congp
from ...util.gSat import gSat
from ...gp.gp1d import gp1d
from ...gp.train import train
from ...base.rollout import rollout
from ...base.propagated import propagated
from ...base.trainDynModel import train_dyn_model
from ...base.learnPolicy import learn_policy
from ...base.applyController import applyController
    from .draw_rollout_pendulum import draw_rollout_pendulum

    # 1. Initialization
    if settings is None:
        settings = get_settings()

    basename = settings.get('basename', 'pendulum_')
    J = settings['J']
    H = settings['H']
    N = settings['N']
    dt = settings['dt']

    mu0 = settings['mu0']
    S0 = settings['S0']
    plant = settings['plant']
    policy = settings['policy']
    cost = settings['cost']
    dynmodel = settings['dynmodel']
    trainOpt = settings['trainOpt']
    opt = settings['opt']
    plotting = settings['plotting']

    # Assign function handles
    plant['dynamics'] = dynamics_pendulum
    plant['prop'] = propagated
    policy['fcn'] = lambda pol, m, s: conCat(congp, gSat, pol, m, s)
    cost['fcn'] = loss_pendulum
    dynmodel['fcn'] = gp1d
    dynmodel['train'] = train

    # Initialize data storage
    x = settings.get('x', np.empty((0, 0)))
    y = settings.get('y', np.empty((0, 0)))
    realCost = settings.get('realCost', [None] * N)
    latent = {}
    M = settings.get('M', [None] * N)
    Sigma = settings.get('Sigma', [None] * N)
    fantasy_mean = settings.get('fantasy_mean', [None] * N)
    fantasy_std = settings.get('fantasy_std', [None] * N)

    # 2. Initial J random rollouts
    for jj in range(J):
        # Random policy (no fcn means random actions)
        rand_policy = {'maxU': policy['maxU']}
        xx, yy, real_cost_traj, latent_traj = rollout(
            _gaussian_single(mu0, S0), rand_policy, H, plant, cost, compute_cost=True
        )
        realCost[jj] = real_cost_traj
        latent[jj] = latent_traj

        if x.size == 0:
            x = xx
            y = yy
        else:
            x = np.vstack([x, xx])
            y = np.vstack([y, yy])

        if plotting['verbosity'] > 0:
            if not plt.fignum_exists(1):
                plt.figure(1)
            else:
                plt.figure(1)
            plt.clf()
            draw_rollout_pendulum(xx, latent, cost, dt, j=None, J=0, H=H, x=x, jj=jj)

    # set up simulated start state (without controls and trig augmentation)
    odei = plant['odei']
    dyno = plant['dyno']
    mu0Sim = mu0[odei].copy()
    S0Sim = S0[np.ix_(odei, odei)].copy()
    mu0Sim = mu0Sim[np.array(dyno)[np.argsort(dyno)] != np.array(odei)[np.argsort(odei)]]
    # Actually: mu0Sim(odei) = mu0; then mu0Sim = mu0Sim(dyno)
    mu0Sim = mu0.copy()
    S0Sim = S0.copy()
    mu0Sim = mu0Sim[np.array(dyno)]
    S0Sim = S0Sim[np.ix_(np.array(dyno), np.array(dyno))]

    # 3. Controlled learning (N iterations)
    for j in range(N):
        dynmodel = train_dyn_model(x, y, dynmodel, policy, plant, trainOpt)

        policy, fX3, M_j, Sigma_j, fm_j, fs_j = learn_policy(
            mu0Sim, S0Sim, dynmodel, policy, plant, cost, H,
            plotting=plotting
        )

        M[j] = M_j
        Sigma[j] = Sigma_j
        fantasy_mean[j] = fm_j
        fantasy_std[j] = fs_j

        x, y, realCost, latent = applyController(
            plant, cost, policy, mu0, S0, H, H, j, J,
            x, y, basename, M, Sigma, dyno,
            plots_verbosity=plotting['verbosity'],
            realCost=realCost, latent=latent,
            rollout_fn=lambda start, pol, hh, p, c: rollout(start, pol, hh, p, c, compute_cost=True)
        )

        print('controlled trial # %d' % (j + 1))

        if plotting['verbosity'] > 0:
            if not plt.fignum_exists(1):
                plt.figure(1)
            else:
                plt.figure(1)
            plt.clf()
            draw_rollout_pendulum(x[-H:, :], latent, cost, dt, j=j, J=J, H=H, x=x,
                                  M=M, Sigma=Sigma)

    # Update settings with results
    settings['x'] = x
    settings['y'] = y
    settings['realCost'] = realCost
    settings['latent'] = latent
    settings['M'] = M
    settings['Sigma'] = Sigma
    settings['fantasy_mean'] = fantasy_mean
    settings['fantasy_std'] = fantasy_std
    settings['policy'] = policy
    settings['dynmodel'] = dynmodel
    settings['mu0Sim'] = mu0Sim
    settings['S0Sim'] = S0Sim

    return settings


def _gaussian_single(mu, sigma):
    """Sample a single point from a Gaussian."""
    D = len(mu)
    L = np.linalg.cholesky(sigma)
    z = np.random.randn(D)
    return mu + L @ z
