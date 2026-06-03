# settings_pendubot.py
# *Summary:* Set up the double-pendulum scenario with two actuators
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

# 1. Define state and important indices

# 1a. Full state representation (including all augmentations)
#  0  dtheta1        angular velocity of inner pendulum
#  1  dtheta2        angular velocity of outer pendulum
#  2  theta1         angle inner pendulum
#  3  theta2         angle outer pendulum
#  4  sin(theta1)    complex representation ...
#  5  cos(theta1)    ... of angle of inner pendulum
#  6  sin(theta2)    complex representation ...
#  7  cos(theta2)    ... of angle of outer pendulum
#  8  u              torque applied to the inner joint

# 1b. Important indices
# odei  indicies for the ode solver
# augi  indicies for variables augmented to the ode variables
# dyno  indicies for the output from the dynamics model and indicies to loss
# angi  indicies for variables treated as angles (using sin/cos representation)
# dyni  indicies for inputs to the dynamics model
# poli  indicies for variables that serve as inputs to the policy
# difi  indicies for training targets that are differences (rather than values)

odei = np.array([0, 1, 2, 3])
augi = np.array([], dtype=int)
dyno = np.array([0, 1, 2, 3])
angi = np.array([2, 3])
dyni = np.array([0, 1, 4, 5, 6, 7])
poli = np.array([0, 1, 4, 5, 6, 7])
difi = np.array([0, 1, 2, 3])


def create_settings():
    """
    Create and return all settings dicts for the pendubot scenario.
    """
    # 2. Set up the scenario
    dt = 0.05                       # [s] sampling time
    T = 3.0                         # [s] prediction time
    H = int(np.ceil(T / dt))        # prediction steps (optimization horizon)
    mu0 = np.array([0, 0, np.pi, np.pi])        # initial state mean
    S0 = np.diag([0.1, 0.1, 0.01, 0.01])**2     # initial state covariance
    N = 40                           # no. of controller optimizations
    J = 1                            # no. of init. training rollouts (of length H)
    K = 1                            # no. of init. states for which we optimize
    nc = 200                         # size of controller training set

    # 3. Set up the plant structure
    from .dynamics_pendubot import dynamics_pendubot
    from ..base.simulate import _zoh

    plant = {}
    plant['dynamics'] = dynamics_pendubot               # dynamics ODE function
    plant['noise'] = np.diag(np.ones(4) * 0.01**2)      # measurement noise
    plant['dt'] = dt
    plant['ctrl'] = 'zoh'        # controller is zero-order-hold
    plant['odei'] = odei          # indices to the varibles for the ode solver
    plant['augi'] = augi          # indices of augmented variables
    plant['angi'] = angi
    plant['poli'] = poli
    plant['dyno'] = dyno
    plant['dyni'] = dyni
    plant['difi'] = difi

    def augment_fn(state):
        return np.array([])

    plant['augment'] = augment_fn

    from ...base.propagate import propagate
    plant['prop'] = propagate     # handle to function that propagates state over time

    # 4. Set up the policy structure
    from ...control.conCat import conCat
    from ...control.congp import congp
    from ...util.gSat import gSat
    from ...util.gTrig import gTrig

    policy = {}
    policy['fcn'] = lambda policy_in, m_in, s_in, compute_deriv=True: \
        conCat(congp, gSat, policy_in, m_in, s_in,
               compute_derivatives=compute_deriv)
    policy['maxU'] = 3.5                               # max. amplitude of torque

    mm, ss, cc = gTrig(mu0, S0, plant['angi'])         # represent angles
    mm_full = np.concatenate([mu0, mm])                # in complex plane
    cc_full = S0 @ cc
    ss_full = np.block([[S0, cc_full],
                         [cc_full.T, ss]])

    from ...util.gaussian import gaussian
    policy_inputs = gaussian(mm_full[poli], ss_full[np.ix_(poli, poli)], nc).T
    policy['p'] = {}
    policy['p']['inputs'] = policy_inputs              # init. location of basis functions
    policy['p']['targets'] = 0.1 * np.random.randn(nc, len(policy['maxU']))  # init. targets
    policy['p']['hyp'] = np.log(np.array([1, 1, 0.7, 0.7, 0.7, 0.7, 1, 0.01]))

    # 5. Set up the cost structure
    from .loss_pendubot import loss_pendubot

    cost = {}
    cost['fcn'] = loss_pendubot                         # cost function
    cost['gamma'] = 1.0                                  # discount factor
    cost['p'] = np.array([0.5, 0.5])                     # lengths of pendulums
    cost['width'] = 0.25                                 # cost function width
    cost['expl'] = 0.0                                   # exploration parameter (UCB)
    cost['angle'] = plant['angi']                       # index of angle (for cost function)
    cost['target'] = np.array([0, 0, 0, 0])              # target state

    # 6. Set up the GP dynamics model structure
    from ...gp.gp1d import gp1d
    from ...gp.train import train_func as train

    dynmodel = {}
    dynmodel['fcn'] = gp1d                       # function for GP predictions
    dynmodel['train'] = train                    # function to train dynamics model
    dynmodel['induce'] = np.zeros((300, 0, 1))   # shared inducing inputs (sparse GP)
    trainOpt = np.array([300, 500])              # defines the max. number of line searches
                                                 # when training the GP dynamics models
                                                 # trainOpt(0): full GP,
                                                 # trainOpt(1): sparse GP (FITC)

    # 7. Parameters for policy optimization
    opt = {}
    opt['length'] = 150                         # max. number of line searches
    opt['MFEPLS'] = 20                          # max. number of function evaluations
                                                 # per line search
    opt['verbosity'] = 1                         # verbosity: specifies how much
                                                 # information is displayed during
                                                 # policy learning. Options: 0-3
    opt['method'] = 'BFGS'                       # optimization method for policy learning.

    # 8. Plotting verbosity
    plotting = {}
    plotting['verbosity'] = 1            # 0: no plots
                                         # 1: some plots
                                         # 2: all plots

    # 9. Some initializations
    x = np.zeros((0, 0))
    y = np.zeros((0, 0))
    fantasy = {'mean': [None] * N, 'std': [None] * N}
    realCost = [None] * N
    M_cell = [None] * N
    Sigma_cell = [None] * N

    return (dt, T, H, mu0, S0, N, J, K, nc,
            plant, policy, cost, dynmodel, trainOpt, opt, plotting,
            x, y, fantasy, realCost, M_cell, Sigma_cell,
            odei, augi, dyno, angi, dyni, poli, difi)
