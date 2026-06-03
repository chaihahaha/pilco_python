# settings_pendulum.py
# *Summary:* Script to set up the pendulum scenario
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-05-24
#
# High-Level Steps
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


def get_settings():
    """Return a dict with all pendulum scenario settings."""

    rng = np.random.RandomState(13)

    # 1. Define state and important indices

    # 1a. Full state representation (including all augmentations)
    #  0  dtheta        angular velocity of pendulum
    #  1  theta         angle of pendulum
    #  2  sin(theta)    complex representation ...
    #  3  cos(theta)    ... of angle of pendulum
    #  4  u             torque applied to the pendulum

    # 1b. Important indices
    # odei  indices for the ode solver
    # augi  indices for variables augmented to the ode variables
    # dyno  indices for the output from the dynamics model and indices to loss
    # angi  indices for variables treated as angles (using sin/cos representation)
    # dyni  indices for inputs to the dynamics model
    # poli  indices for variables that serve as inputs to the policy
    # difi  indices for training targets that are differences (rather than values)

    odei = [0, 1]              # variables for the ODE solver
    augi = []                  # variables to be augmented
    dyno = [0, 1]              # variables to be predicted (and known to loss)
    angi = [1]                 # angle variables  (0-indexed, was [2] in MATLAB)
    dyni = [0, 2, 3]           # variables that serve as inputs to the dynamics GP
    poli = [0, 2, 3]           # variables that serve as inputs to the policy
    difi = [0, 1]              # variables that are learned via differences

    # 2. Set up the scenario
    dt = 0.1                       # [s] sampling time
    T = 4.0                        # [s] prediction time
    H = int(np.ceil(T / dt))       # prediction steps (optimization horizon)
    mu0 = np.array([0.0, 0.0])     # initial state mean
    S0 = 0.01 * np.eye(2)          # initial state variance
    N = 10                         # number of policy optimizations
    J = 1                          # no. of initial training rollouts (of length H)
    K = 1                          # number of initial states for which we optimize
    nc = 20                        # size of controller training set

    # 3. Set up the plant structure
    plant = {}
    plant['dynamics'] = None       # will be set to dynamics_pendulum
    plant['dt'] = dt
    plant['ctrl'] = 'zoh'          # controller is zero-order-hold
    plant['odei'] = np.array(odei)
    plant['augi'] = np.array(augi)
    plant['angi'] = np.array(angi)
    plant['poli'] = np.array(poli)
    plant['dyno'] = np.array(dyno)
    plant['dyni'] = np.array(dyni)
    plant['difi'] = np.array(difi)
    plant['noise'] = np.diag(np.array([0.1**2, 0.01**2]))
    # plant['prop'] will be set to propagated

    # 4. Set up the policy structure
    # Represent angles in complex plane
    from ...util.gTrig import gTrig
    mm, ss, cc = gTrig(mu0, S0, plant['angi'], compute_derivatives=False)
    mm_full = np.concatenate([mu0, mm])
    ss_full = np.block([
        [S0,       cc],
        [cc.T,     ss]
    ])
    poli_arr = np.array(poli)

    policy = {}
    policy['maxU'] = np.array([2.5])               # max. amplitude of torque

    policy_inputs = _gaussian(mm_full[poli_arr], ss_full[np.ix_(poli_arr, poli_arr)], nc, rng)
    policy['p'] = {}
    policy['p']['inputs'] = policy_inputs.T        # init. location of basis functions (N x D)
    policy['p']['targets'] = 0.1 * rng.randn(nc, len(policy['maxU']))
    # init. policy targets (close to zero)
    policy['p']['hyp'] = np.log(np.array([1.0, 0.7, 0.7, 1.0, 0.01]))  # initialize hyper-parameters

    # 5. Set up the cost structure
    cost = {}
    cost['gamma'] = 1.0                            # discount factor
    cost['p'] = 1.0                                # length of pendulum
    cost['width'] = 0.5                            # cost function width
    cost['expl'] = 0.0                             # exploration parameter
    cost['angle'] = plant['angi'].copy()           # angle variables in cost (0-indexed)
    cost['target'] = np.array([0.0, np.pi])        # target state

    # 6. Set up the GP dynamics model structure
    dynmodel = {}
    # dynmodel.fcn and dynmodel.train will be set externally
    dynmodel['induce'] = np.zeros((300, 0, 1))
    trainOpt = [300, 500]
    # defines the max. number of line searches when training the GP dynamics models
    # trainOpt[0]: full GP, trainOpt[1]: sparse GP (FITC)

    # 7. Parameters for policy optimization
    opt = {}
    opt['length'] = 75                        # max. number of line searches
    opt['MFEPLS'] = 30                        # max. number of function evaluations per line search
    opt['verbosity'] = 1                      # verbosity: specifies how much information is displayed during policy learning. Options: 0-3
    opt['method'] = 'BFGS'                    # optimization algorithm. Options: 'BFGS' (default), 'LBFGS', 'CG'

    # 8. Plotting verbosity
    plotting = {}
    plotting['verbosity'] = 1                 # 0: no plots, 1: some plots, 2: all plots

    # 9. Some initializations
    x = np.empty((0, 0))
    y = np.empty((0, 0))
    fantasy_mean = [None] * N
    fantasy_std = [None] * N
    realCost = [None] * N
    M = [None] * N
    Sigma = [None] * N

    settings = {
        'dt': dt, 'T': T, 'H': H,
        'mu0': mu0, 'S0': S0,
        'N': N, 'J': J, 'K': K, 'nc': nc,
        'odei': odei, 'augi': augi, 'dyno': dyno, 'angi': angi,
        'dyni': dyni, 'poli': poli, 'difi': difi,
        'plant': plant,
        'policy': policy,
        'cost': cost,
        'dynmodel': dynmodel,
        'trainOpt': trainOpt,
        'opt': opt,
        'plotting': plotting,
        'x': x, 'y': y,
        'fantasy_mean': fantasy_mean, 'fantasy_std': fantasy_std,
        'realCost': realCost, 'M': M, 'Sigma': Sigma,
        'basename': 'pendulum_',
    }
    return settings


def _gaussian(mu, sigma, n, rng):
    """Sample n points from a multivariate Gaussian N(mu, sigma).

    Parameters
    ----------
    mu : ndarray (D,)
        Mean vector
    sigma : ndarray (D, D)
        Covariance matrix
    n : int
        Number of samples
    rng : numpy.random.RandomState
        Random number generator for reproducibility

    Returns
    -------
    x : ndarray (D, n)
        Sampled points
    """
    D = len(mu)
    L = np.linalg.cholesky(sigma)
    Z = rng.randn(D, n)
    x = mu.reshape(-1, 1) + L @ Z
    return x
