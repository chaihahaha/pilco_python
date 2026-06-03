# cartDouble_learn.py
# *Summary:* Script to learn a controller for the cart-doube-pendulum
# swingup
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
#
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
import matplotlib.pyplot as plt
from .settings_cdp import *
from .dynamics_cdp import dynamics_cdp
from .loss_cdp import loss_cdp

from ..gp.gp1d import gp1d
from ..gp.train import train
from ..control.conCat import conCat
from ..control.congp import congp
from ..util.gSat import gSat
from ..util.gTrig import gTrig
from ..util.gaussian import gaussian
from ..base.rollout import rollout
from ..base.trainDynModel import train_dyn_model
from ..base.learnPolicy import learn_policy
from ..base.applyController import applyController
from ..base.propagated import propagated


def cartDouble_learn():
    ## Code

    # 1. Initialization
    basename = 'CartDoublePend_'  # filename used for saving data

    # Set up the plant structure with actual function references
    plant['dynamics'] = dynamics_cdp
    plant['prop'] = propagated

    # Set up the policy structure with actual function references
    # policy.fcn = conCat(congp, gSat, policy, m, s)
    policy['fcn'] = conCat_concat  # will be set below

    # Set up the cost with actual function reference
    cost['fcn'] = loss_cdp

    # Set up dynamics model with actual function references
    dynmodel['fcn'] = gp1d
    dynmodel['train'] = train

    # represent angles in complex plane for policy initialization
    mm, ss, cc = gTrig(mu0, S0, plant['angi'], compute_derivatives=False)
    mm_full = np.concatenate([mu0.ravel(), mm.ravel()])
    cc_full = S0 @ cc
    ss_full = np.block([[S0, cc_full],
                        [cc_full.T, ss]])

    policy['p'] = {}
    policy['p']['inputs'] = gaussian(mm_full[poli], ss_full[np.ix_(poli, poli)], nc).T
    policy['p']['targets'] = 0.1*np.random.randn(nc, len(policy['maxU']))
    policy['p']['hyp'] = np.log(np.array([1, 1, 1, 1, 0.7, 0.7, 0.7, 0.7, 1, 0.01]))

    # 2. Initial J random rollouts
    # Define the conCat wrapper
    def policy_fcn_concat(policy_var, m_val, s_val, compute_derivatives=False):
        return conCat(congp, gSat, policy_var, m_val, s_val, compute_derivatives=compute_derivatives)

    policy['fcn'] = policy_fcn_concat

    x_data = np.empty((0, 0))
    y_data = np.empty((0, 0))
    latent = {}
    realCost = {}

    for jj in range(J):
        xx, yy, rCost, lat = rollout(
            gaussian(mu0, S0),
            {'maxU': policy['maxU']},
            H, plant, cost,
            compute_cost=False)

        if x_data.size == 0:
            x_data = xx; y_data = yy
        else:
            x_data = np.vstack([x_data, xx]); y_data = np.vstack([y_data, yy])

        realCost[jj] = rCost
        latent[jj] = lat

        if plotting['verbosity'] > 0:
            plt.figure(1)
            from .draw_rollout_cdp import draw_rollout_cdp
            draw_rollout_cdp_impl(xx, latent, M, Sigma, cost, H, dt, None, J, jj)

    mu0Sim = mu0[dyno].copy()
    S0Sim = S0[np.ix_(dyno, dyno)].copy()

    # 3. Controlled learning (N iterations)
    for j in range(N):
        # train (GP) dynamics model
        dynmodel_result = train_dyn_model(x_data, y_data, dynmodel, policy, plant, trainOpt)
        dynmodel.update(dynmodel_result)

        # learn policy
        policy_result = learn_policy(mu0Sim, S0Sim, dynmodel, policy, plant, cost, H,
                                       plotting=plotting)
        policy_new, fX3, M_pred, Sigma_pred, f_mean, f_std = policy_result
        policy = policy_new

        # store for this trial
        M[j] = M_pred
        Sigma[j] = Sigma_pred
        fantasy_mean[j] = f_mean
        fantasy_std[j] = f_std

        # apply controller to system
        x_data, y_data, realCost, latent = applyController(
            plant, cost, policy, mu0, S0, H, maxH, j, J,
            x_data, y_data, basename, M, Sigma, dyno,
            plots_verbosity=plotting['verbosity'],
            realCost=realCost, latent=latent,
            rollout_fn=rollout)

        print(f'controlled trial # {j}')

        if plotting['verbosity'] > 0:
            plt.figure(1)
            from .draw_rollout_cdp import draw_rollout_cdp
            draw_rollout_cdp_impl(x_data, latent, M, Sigma, cost, H, dt, j, J, 0)

    return policy, realCost, latent, dynmodel


def draw_rollout_cdp_impl(xx, latent, M, Sigma, cost, H, dt, j, J, jj=0):
    """Helper to call draw_rollout_cdp"""
    from .draw_rollout_cdp import draw_rollout_cdp
    draw_rollout_cdp(xx, latent, M, Sigma, cost, H, dt, j, J, jj)


def conCat_concat(policy, m, s, compute_derivatives=False):
    """Wrapper for conCat(congp, gSat, policy, m, s)"""
    return conCat(congp, gSat, policy, m, s, compute_derivatives=compute_derivatives)


if __name__ == '__main__':
    cartDouble_learn()
    plt.show()
