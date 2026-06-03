# settings_dp.py
# *Summary:* Script set up the double-pendulum scenario with two actuators
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


## Code

import numpy as np
from pilco_python.util.gTrig import gTrig
from pilco_python.util.gaussian import gaussian
from pilco_python.util.gSat import gSat
from pilco_python.control.conCat import conCat
from pilco_python.control.congp import congp
from pilco_python.base.propagated import propagated
from pilco_python.gp.gp1d import gp1d
from pilco_python.gp.train import train


def settings_dp(rng=None):
    """
    Set up the double-pendulum scenario with two actuators.
    Returns a dict with: plant, policy, cost, dynmodel, opt, plotting,
    mu0, S0, H, N, J, K, nc, dt, T, realCost, fantasy, M_cell, Sigma_cell.
    """

    if rng is None:
        rng = np.random.RandomState(1)

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
    #  8  u1             torque applied to the inner joint
    #  9  u2             torque applied to the outer joint

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


    # 2. Set up the scenario
    dt = 0.1                          # [s] sampling time
    T = 3.0                           # [s] prediction horizon
    H = int(np.ceil(T / dt))          # prediction steps (optimization horizon)
    mu0 = np.array([0, 0, np.pi, np.pi])        # initial state mean
    S0 = np.diag(np.array([0.1, 0.1, 0.01, 0.01])**2)  # initial state covariance
    N = 20                             # no. of controller optimizations
    J = 1                              # no. of initial training rollouts (length H)
    K = 1                              # no. of initial states for which we optimize
    nc = 100                           # size of controller training set

    # 3. Set up the plant structure
    plant = {
        'dynamics': _dynamics_dp_wrapper,
        'noise': np.diag(np.ones(4) * 0.01**2),              # measurement noise
        'dt': dt,
        'ctrl': 'zoh',                   # controler is zero-order-hold
        'odei': odei,                     # indices to the varibles for the ode solver
        'augi': augi,                     # indices of augmented variables
        'angi': angi,
        'poli': poli,
        'dyno': dyno,
        'dyni': dyni,
        'difi': difi,
        'prop': propagated,    # handle to function that propagates state over time
    }


    # 4. Set up the policy structure
    _mm, _ss, _cc = gTrig(mu0, S0, plant['angi'], compute_derivatives=False)  # represent angles
    mm = np.concatenate([mu0, _mm])
    cc = S0 @ _cc
    ss = np.block([[S0, cc], [cc.T, _ss]])                       # in complex plane

    policy_maxU = np.array([2, 2])                               # max. amplitude of torques

    policy_inputs = gaussian(mm[poli], ss[np.ix_(poli, poli)], nc).T  # init. location of
                                                                      # basis functions
    policy_targets = 0.1 * rng.randn(nc, len(policy_maxU))       # init. policy targets
                                                                  # (close to zero)
    policy_hyp = np.tile(np.log(np.array([1, 1, 0.7, 0.7, 0.7, 0.7, 1, 0.01])), (2, 1)).T.flatten('F')  # initialize policy hyper-parameters

    policy = {
        'fcn': lambda policy, m, s, compute_derivatives=False: conCat(congp, gSat, policy, m, s, compute_derivatives=compute_derivatives),
        'maxU': policy_maxU,
        'p': {
            'inputs': policy_inputs,
            'targets': policy_targets,
            'hyp': policy_hyp,
        }
    }


    # 5. Set up the cost structure
    from pilco_python.scenarios.doublePendulum.loss_dp import loss_dp
    cost = {
        'fcn': loss_dp,                               # cost function
        'gamma': 1,                                   # discount factor
        'p': np.array([0.5, 0.5]),                    # lengths of pendulums
        'width': 0.5,                                 # cost function width
        'expl': 0,                                    # exploration parameter (UCB)
        'angle': plant['angi'],                       # index of angle (for cost function)
        'target': np.array([0, 0, 0, 0]),             # target state
    }

    # 6. Set up the GP dynamics model structure
    dynmodel = {
        'fcn': gp1d,                    # function for GP predictions
        'train': train,                 # function to train dynamics model
        'induce': np.zeros((300, 0, 1)),  # shared inducing inputs (sparse GP)
    }
    trainOpt = np.array([300, 500])     # defines the max. number of line searches
                                        # when training the GP dynamics models
                                        # trainOpt[0]: full GP,
                                        # trainOpt[1]: sparse GP (FITC)

    # 7. Parameters for policy optimization
    opt = {
        'length': 150,                     # max. number of line searches
        'MFEPLS': 30,                      # max. number of function evaluations
                                           # per line search
        'verbosity': 1,                    # verbosity: specifies how much
                                           # information is displayed during
                                           # policy learning. Options: 0-3
    }

    # 8. Plotting verbosity
    plotting = {
        'verbosity': 1,            # 0: no plots
                                   # 1: some plots
                                   # 2: all plots
    }

    # 9. Some initializations
    x = np.empty((0, 0))
    y = np.empty((0, 0))
    fantasy = {'mean': [None] * N, 'std': [None] * N}
    realCost = [None] * N
    M_cell = [None] * N
    Sigma_cell = [None] * N

    result = {
        'plant': plant,
        'policy': policy,
        'cost': cost,
        'dynmodel': dynmodel,
        'trainOpt': trainOpt,
        'opt': opt,
        'plotting': plotting,
        'mu0': mu0,
        'S0': S0,
        'dt': dt,
        'T': T,
        'H': H,
        'N': N,
        'J': J,
        'K': K,
        'nc': nc,
        'x': x,
        'y': y,
        'fantasy': fantasy,
        'realCost': realCost,
        'M_cell': M_cell,
        'Sigma_cell': Sigma_cell,
        'rng': rng,
    }

    return result


def _dynamics_dp_wrapper(t, z, u_vals):
    """Wrapper to call dynamics_dp with the simulate.py convention (t, z, u_vals)."""
    from pilco_python.scenarios.doublePendulum.dynamics_dp import dynamics_dp
    return dynamics_dp(t, z, u_vals, compute_derivative=True)
