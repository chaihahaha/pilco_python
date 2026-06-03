# settings_cdp.py
# *Summary:* Script set up the cart-double-pendulum scenario
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-27
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
#
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np
import warnings

# 1. Define state and important indices

# 1a. Full state representation (including all augmentations)
#  0   x             position of the cart
#  1   dx            velocity of the cart
#  2   dtheta1       angular velocity of inner pendulum
#  3   dtheta2       angular velocity of outer pendulum
#  4   theta1        angle of inner pendulum
#  5   theta2        angle of outer pendulum
#  6   sin(theta1)   complex representation ...
#  7   cos(theta1)   ... of theta1
#  8   sin(theta2)   complex representation ...
#  9   cos(theta2)   ... of theta2
#  10  u             force that can be applied at cart

# 1b. Important indices (0-indexed)
odei = np.array([0, 1, 2, 3, 4, 5])
augi = np.array([], dtype=int)
dyno = np.array([0, 1, 2, 3, 4, 5])
angi = np.array([4, 5])
dyni = np.array([0, 1, 2, 3, 6, 7, 8, 9])
poli = np.array([0, 1, 2, 3, 6, 7, 8, 9])
difi = np.array([0, 1, 2, 3, 4, 5])

# 2. Set up the scenario
dt = 0.05               # [s] sampling time
T = 5.0                 # [s] prediction time
H = int(np.ceil(T/dt))  # prediction steps (optimization horizon)
maxH = H                # max pred horizon
nc = 200                # size of controller training set
s_diag = np.array([0.1, 0.1, 0.1, 0.1, 0.01, 0.01])**2  # initial state variances
S0 = np.diag(s_diag)    # initial state covariance matrix
mu0 = np.array([0, 0, 0, 0, np.pi, np.pi])  # initial state mean
N = 40                  # number of policy searches
J = 1                   # J initial (random) trajectories, each of length H
K = 1                   # number of initial states for which we optimize

# 3. Set up the plant structure
# plant is a dict

plant = {}
plant['dynamics'] = None       # will be set to dynamics_cdp
plant['noise'] = np.diag(np.ones(6)*0.01**2)  # measurement noise
plant['dt'] = dt
plant['ctrl'] = 'zoh'          # controller is zero order hold
plant['odei'] = odei            # indices to the variables for the ode solver
plant['augi'] = augi            # indices of augmented variables
plant['angi'] = angi
plant['poli'] = poli
plant['dyno'] = dyno
plant['dyni'] = dyni
plant['difi'] = difi
plant['prop'] = None            # will be set to propagated

# 4. Set up the policy structure
policy = {}
policy['fcn'] = None            # will be set to conCat(congp, gSat,...)
policy['maxU'] = np.array([20])  # max. amplitude of control

# 5. Set up the cost structure
cost = {}
cost['fcn'] = None              # will be set to loss_cdp
cost['gamma'] = 1               # discount factor
cost['p'] = np.array([1, 1])     # lengths of the links
cost['width'] = 0.5             # cost function width
cost['expl'] = 0                # exploration parameter
cost['angle'] = angi            # angle variables in cost
cost['target'] = np.zeros(6)    # target state

# 6. Set up the GP dynamics model structure
dynmodel = {}
dynmodel['fcn'] = None          # will be set to gp1d
dynmodel['train'] = None         # will be set to train
dynmodel['induce'] = np.zeros((400, 0, 1))  # shared inducing inputs (sparse GP)
trainOpt = np.array([300, 500])  # defines the max. number of line searches
                                  # when training the GP dynamics models
                                  # trainOpt[0]: full GP, trainOpt[1]: sparse GP (FITC)

# 7. Parameters for policy optimization
opt = {}
opt['length'] = 150              # max. number of line searches
opt['MFEPLS'] = 30               # max. number of function evaluations per line search
opt['verbosity'] = 1             # verbosity: 0-3
opt['method'] = 'BFGS'           # optimization algorithm: BFGS, LBFGS, CG

# 8. Plotting verbosity
plotting = {}
plotting['verbosity'] = 1        # 0: no plots, 1: some plots, 2: all plots

# 9. Initialize various variables
x = None; y = None
fantasy_mean = [None]*N; fantasy_std = [None]*N
realCost = [None]*N; M = [None]*N; Sigma = [None]*N
