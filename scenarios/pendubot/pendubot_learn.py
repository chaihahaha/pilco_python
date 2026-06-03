# pendubot_learn.py
# *Summary:* Script to learn a controller for the pendubot
# swingup (a double pendulum with the inner joint actuated)
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
from .settings_pendubot import create_settings
from ...util.gaussian import gaussian


def pendubot_learn():
    ## Code

    # 1. Initialization
    (dt, T, H, mu0, S0, N, J, K, nc,
     plant, policy, cost, dynmodel, trainOpt, opt, plotting,
     x, y, fantasy, realCost, M_cell, Sigma_cell,
     odei, augi, dyno, angi, dyni, poli, difi) = create_settings()

    basename = 'pendubot_'       # filename used for saving data

    # 2. Initial J random rollouts
    from ...base.rollout import rollout

    for jj in range(J):
        xx, yy, realCost_cur, latent_cur = \
            rollout(gaussian(mu0, S0), {'maxU': policy['maxU']}, H, plant, cost)
        realCost[jj] = realCost_cur
        latent = {jj: latent_cur}
        if x.size == 0:
            x = xx; y = yy
        else:
            x = np.vstack([x, xx])
            y = np.vstack([y, yy])

        if plotting['verbosity'] > 0:      # visualization of trajectory
            plt = _ensure_plt()
            plt.figure(1)
            plt.clf()
            from .draw_rollout_pendubot import draw_rollout_pendubot
            draw_rollout_pendubot(
                0, J, xx, latent, M_cell, Sigma_cell, cost, H, dt, x)

    # 4D state to augmented state used for dynamics model
    mu0Sim = np.zeros((len(odei) + len(augi), J))
    mu0Sim[np.ix_(odei, [0])] = mu0.reshape(-1, 1)
    S0Sim = np.zeros((len(odei) + len(augi), len(odei) + len(augi)))
    S0Sim[np.ix_(odei, odei)] = S0
    mu0Sim = mu0Sim[np.ix_(dyno, [0])]
    S0Sim = S0Sim[np.ix_(dyno, dyno)]

    # 3. Controlled learning (N iterations)
    from ...base.trainDynModel import trainDynModel
    from ...base.learnPolicy import learnPolicy
    from ...base.applyController import applyController

    for j_idx in range(N):
        dynmodel = trainDynModel(dynmodel, x, y, plant, trainOpt)   # train (GP) dynamics model
        policy = learnPolicy(policy, plant, dynmodel, cost, opt)    # learn policy
        x, y, realCost, latent = applyController(                   # apply controller to system
            plant, cost, policy, mu0Sim, S0Sim,
            H, H, j_idx, J, x, y, basename,
            M_cell, Sigma_cell, dyno,
            plots_verbosity=plotting['verbosity'],
            realCost=realCost, latent=latent,
            rollout_fn=None)  # NOTE: needs actual rollout function
        print('controlled trial # %d' % j_idx)
        if plotting['verbosity'] > 0:      # visualization of trajectory
            plt = _ensure_plt()
            plt.figure(1)
            plt.clf()
            from .draw_rollout_pendubot import draw_rollout_pendubot
            draw_rollout_pendubot(
                j_idx, J, None, latent, M_cell, Sigma_cell, cost, H, dt, x)

    return (x, y, realCost, policy, dynmodel, cost, plant)


def _ensure_plt():
    import matplotlib.pyplot as plt
    return plt
