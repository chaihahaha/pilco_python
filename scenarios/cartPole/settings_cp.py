# settings_cp.py
# *Summary:* Script to set up the cart-pole scenario
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-24
#
## High-Level Steps
# # Define state and important indices
# # Set up scenario
# # Set up the plant structure
# # Set up the policy structure
# # Set up the cost structure
# # Set up the GP dynamics model structure
# # Parameters for policy optimization
# # Plotting verbosity
# # Some array initializations

import numpy as np
from .dynamics_cp import dynamics_cp
from .loss_cp import loss_cp


def generate_initial_rollout(plant, policy, cost, H, mu0, S0):
    """Generate initial random rollout matching MATLAB settings_cp.m logic:
       rand('seed',1); randn('seed',1);
       rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost)
    """
    from pilco_python.util.gaussian import gaussian
    from pilco_python.base.rollout import rollout

    np.random.seed(1)

    plant = plant.copy()
    plant['ctrl'] = 'zoh'  # required by simulate()

    start_state = gaussian(mu0, S0)
    xx, yy, _, _ = rollout(start_state, policy, H, plant, cost, compute_cost=False)
    return xx, yy


def get_initial_rollout_data(plant, policy, cost, H, mu0, S0):
    """Return initial rollout data generated deterministically with seed 1."""
    return generate_initial_rollout(plant, policy, cost, H, mu0, S0)


def define_settings():
    # Code

    # 1. Define state and important indices

    # 1a. Full state representation (including all augmentations)
    #
    #  0  x          cart position
    #  1  v          cart velocity
    #  2  dtheta     angular velocity
    #  3  theta      angle of the pendulum
    #  4  sin(theta) complex representation ...
    #  5  cos(theta) of theta
    #  6  u          force applied to cart
    #

    # 1b. Important indices (0-based)
    # odei  indicies for the ode solver
    # augi  indicies for variables augmented to the ode variables
    # dyno  indicies for the output from the dynamics model and indicies to loss
    # angi  indicies for variables treated as angles (using sin/cos representation)
    # dyni  indicies for inputs to the dynamics model
    # poli  indicies for the inputs to the policy
    # difi  indicies for training targets that are differences (rather than values)

    odei = np.array([0, 1, 2, 3])        # varibles for the ode solver
    augi = np.array([], dtype=int)       # variables to be augmented
    dyno = np.array([0, 1, 2, 3])        # variables to be predicted (and known to loss)
    angi = np.array([3])                 # angle variables
    dyni = np.array([0, 1, 2, 4, 5])     # variables that serve as inputs to the dynamics GP
    poli = np.array([0, 1, 2, 4, 5])     # variables that serve as inputs to the policy
    difi = np.array([0, 1, 2, 3])        # variables that are learned via differences

    # 2. Set up the scenario
    dt = 0.10                            # [s] sampling time
    T = 4.0                              # [s] initial prediction horizon time
    H = int(np.ceil(T / dt))             # prediction steps (optimization horizon)
    mu0 = np.array([0.0, 0.0, 0.0, 0.0])  # initial state mean
    S0 = np.diag([0.1, 0.1, 0.1, 0.1])**2  # initial state covariance
    N = 15   # number controller optimizations
    J = 1    # initial J trajectories of length H
    K = 1    # no. of initial states for which we optimize
    nc = 10  # number of controller basis functions

    # 3. Plant structure
    plant = {
        'dynamics': dynamics_cp,
        'noise': np.diag(np.ones(4) * 0.01**2),
        'dt': dt,
        'ctrl': None,                     # zoh placeholder; implemented in rollout/simulate
        'odei': odei,
        'augi': augi,
        'angi': angi,
        'poli': poli,
        'dyno': dyno,
        'dyni': dyni,
        'difi': difi,
        'prop': None,                     # placeholder; propagated from base module
    }

    # 4. Policy structure
    # Matching MATLAB settings_cp.m:
    #   rand('seed',1); randn('seed',1);
    #   [mm ss cc] = gTrig(mu0, S0, plant.angi);
    #   mm = [mu0; mm]; cc = S0*cc; ss = [S0 cc; cc' ss];
    #   policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)';
    #   policy.p.targets = 0.1*randn(nc, length(policy.maxU));
    #   policy.p.hyp = log([1 1 1 0.7 0.7 1 0.01])';
    from pilco_python.util.gTrig import gTrig

    np.random.seed(1)

    mm, ss, cc = gTrig(mu0, S0, plant['angi'], compute_derivatives=False)
    mm_full = np.concatenate([mu0, mm])
    cc_full = S0 @ cc
    ss_full = np.vstack([
        np.hstack([S0, cc_full]),
        np.hstack([cc_full.T, ss]),
    ])

    mm_poli = mm_full[poli]
    ss_poli = ss_full[np.ix_(poli, poli)]

    # policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)'
    from pilco_python.util.gaussian import gaussian
    inputs = gaussian(mm_poli, ss_poli, nc).T  # (D x n) -> (nc, D) = (10, 5)

    # policy.p.targets = 0.1*randn(nc, length(policy.maxU))
    maxU = np.array([10.0])
    targets = 0.1 * np.random.randn(nc, len(maxU))
    # policy.p.hyp = log([1 1 1 0.7 0.7 1 0.01])'
    hyp = np.log(np.array([1.0, 1.0, 1.0, 0.7, 0.7, 1.0, 0.01]))

    policy = {
        'fcn': None,                      # placeholder; conCat(@congp, @gSat, ...)
        'maxU': maxU,
        'p': {
            'inputs': inputs,
            'targets': targets,
            'hyp': hyp,
        }
    }

    # 5. Set up the cost structure
    cost = {
        'fcn': loss_cp,
        'gamma': 1,
        'p': 0.5,
        'width': 0.25,
        'expl': 0.0,
        'angle': np.array([3]),  # plant.angi
        'target': np.array([0.0, 0.0, 0.0, np.pi]),
    }

    # 6. Dynamics model structure
    dynmodel = {
        'fcn': None,               # placeholder; gp1d
        'train': None,             # placeholder; train
        'induce': np.zeros((300, 0, 1)),
    }
    trainOpt = [300, 500]          # max. number of line searches for GP training

    # 7. Parameters for policy optimization
    opt = {
        'length': 150,
        'MFEPLS': 30,
        'verbosity': 1,
    }

    # 8. Plotting verbosity
    plotting = {'verbosity': 1}    # 0: no plots, 1: some plots, 2: all plots

    # 9. Some initializations
    x = np.zeros((0, len(dyno)))
    y = np.zeros((0, len(dyno)))
    fantasy_mean = [None] * N
    fantasy_std = [None] * N
    realCost = [None] * N
    M = [None] * N
    Sigma = [None] * N

    settings = {
        'odei': odei, 'augi': augi, 'dyno': dyno, 'angi': angi,
        'dyni': dyni, 'poli': poli, 'difi': difi,
        'dt': dt, 'T': T, 'H': H, 'mu0': mu0, 'S0': S0,
        'N': N, 'J': J, 'K': K, 'nc': nc,
        'plant': plant, 'policy': policy, 'cost': cost,
        'dynmodel': dynmodel, 'trainOpt': trainOpt, 'opt': opt,
        'plotting': plotting,
        'x': x, 'y': y,
        'fantasy_mean': fantasy_mean, 'fantasy_std': fantasy_std,
        'realCost': realCost, 'M': M, 'Sigma': Sigma,
        'mu0Sim': mu0[dyno],
        'S0Sim': S0[np.ix_(dyno, dyno)],
    }

    return settings
