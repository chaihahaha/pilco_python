# DoublePend_learn.py
# *Summary:* Script to learn a controller for the double-pendulum
# swingup with two actuated joints
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
from pilco_python.scenarios.doublePendulum.settings_dp import settings_dp
from pilco_python.base.rollout import rollout
from pilco_python.base.trainDynModel import trainDynModel
from pilco_python.base.learnPolicy import learnPolicy
from pilco_python.base.applyController import applyController


def DoublePend_learn(plots_verbosity=None, rng=None):
    """
    Run the double pendulum learning loop.

    Parameters
    ----------
    plots_verbosity : int, optional
        Override plotting verbosity (0: none, 1: some, 2: all)
    rng : numpy.random.RandomState, optional
        Random state for reproducibility

    Returns
    -------
    cfg : dict
        The settings dictionary with results populated.
    """
    ## Code

    # 1. Initialization
    cfg = settings_dp(rng=rng)
    basename = 'doublepend_'     # filename used for saving data

    mu0 = cfg['mu0']
    S0 = cfg['S0']
    plant = cfg['plant']
    policy = cfg['policy']
    cost = cfg['cost']
    dynmodel = cfg['dynmodel']
    H = cfg['H']
    J = cfg['J']
    N = cfg['N']
    trainOpt = cfg['trainOpt']
    opt = cfg['opt']
    plotting = cfg['plotting']
    dt = cfg['dt']

    if plots_verbosity is not None:
        plotting['verbosity'] = plots_verbosity

    x = cfg['x']
    y = cfg['y']
    latent = {}
    realCost = {}

    # 2. Initial J random rollouts
    for jj in range(J):
        xx, yy, rc, lat = rollout(
            np.random.multivariate_normal(mu0, S0),
            {'maxU': policy['maxU']}, H, plant, cost)
        if x.size == 0:
            x = xx
        else:
            x = np.vstack([x, xx])
        if y.size == 0:
            y = yy
        else:
            y = np.vstack([y, yy])
        realCost[jj] = rc
        latent[jj] = lat

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(1)
            plt.clf()
            from pilco_python.scenarios.doublePendulum.draw_rollout_dp import draw_rollout_dp
            draw_rollout_dp(xx, latent, cost, dt, H, None, None, J=0, j=None, jj=jj)

    odei = plant['odei']
    mu0Sim = np.zeros(len(odei))
    mu0Sim[odei] = mu0
    S0Sim = np.zeros((len(odei), len(odei)))
    S0Sim[np.ix_(odei, odei)] = S0
    dyno = plant['dyno']
    mu0Sim = mu0Sim[dyno]
    S0Sim = S0Sim[np.ix_(dyno, dyno)]

    # 3. Controlled learning (N iterations)
    for j in range(N):
        cfg['dynmodel'] = trainDynModel
        cfg['plant'] = plant
        cfg['policy'] = policy
        cfg['cost'] = cost
        cfg['x'] = x
        cfg['y'] = y

        dynmodel = trainDynModel(dynmodel, plant, trainOpt, x, y, opt)
        policy, M_cell_j, Sigma_cell_j = learnPolicy(dynmodel, plant, policy, cost,
                                                      mu0Sim, S0Sim, H, opt)
        x, y, rc, lat = applyController(plant, cost, policy, mu0, S0, H, H,
                                         j, J, x, y, basename,
                                         M_cell_j, Sigma_cell_j, dyno,
                                         plots_verbosity=plotting['verbosity'],
                                         realCost=realCost, latent=latent)

        realCost[j + J] = rc
        latent[j] = lat
        print('controlled trial # %d' % (j + 1))

        if plotting['verbosity'] > 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(1)
            plt.clf()
            from pilco_python.scenarios.doublePendulum.draw_rollout_dp import draw_rollout_dp
            draw_rollout_dp(x, latent, cost, dt, H, M_cell_j, Sigma_cell_j, J=J, j=j, jj=0)

    cfg['x'] = x
    cfg['y'] = y
    cfg['latent'] = latent
    cfg['realCost'] = realCost

    return cfg
