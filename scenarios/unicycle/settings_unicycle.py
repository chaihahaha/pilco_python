# settings_unicycle.py
# *Summary:* Script to set up the unicycle scenario
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-04-02
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

# 1. Define state and important indices
#
# 1a. Full state representation (including all augmentations) - 0-indexed
#  0  dx      x velocity
#  1  dy      y velocity
#  2  dxc     x velocity of origin (unicycle coordinates)
#  3  dyc     y velocity of origin (unicycle coordinates)
#  4  dtheta  roll angular velocity
#  5  dphi    yaw angular velocity
#  6  dpsiw   wheel angular velocity
#  7  dpsif   pitch angular velocity
#  8  dpsit   turn table angular velocity
#  9  x       x position
# 10  y       y position
# 11  xc      x position of origin (unicycle coordinates)
# 12  yc      y position of origin (unicycle coordinates)
# 13  theta   roll angle
# 14  phi     yaw angle
# 15  psiw    wheel angle
# 16  psif    pitch angle
# 17  psit    turn table angle
# 18  ct      control torque for turn table
# 19  cw      control torque for wheel
#
# 1b. Important indices (0-based)
# odei  indicies for the ode solver
# augi  indicies for variables augmented to the ode variables
# dyno  indicies for the output from the dynamics model and indicies to loss
# angi  indicies for variables treated as angles (using sin/cos representation)
# dyni  indicies for inputs to the dynamics model
# poli  indicies for variables that serve as inputs to the policy
# difi  indicies for training targets that are differences (rather than values)

odei = np.array([4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17])
augi = np.array([0, 1, 2, 3, 11, 12])
dyno = np.array([4, 5, 6, 7, 8, 11, 12, 13, 14, 16])
angi = np.array([], dtype=int)
dyni = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])
poli = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
difi = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 2. Set up the scenario
dt = 0.15                    # [s] sampling time
T = 10.0                     # [s] initial prediction horizon time
H = int(np.ceil(T / dt))     # prediction steps (optimization horizon)
maxH = int(np.ceil(10.0 / dt))  # max pred horizon
s = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]) ** 2
S0 = np.diag(s)              # initial state variance
mu0 = np.zeros(len(odei))    # initial state mean
N = 40                       # number controller optimizations
J = 10                       # initial J trajectories of length H
K = 1                        # number of initial states for which we optimize

# 3. Set up the plant structure
plant = {}
plant['dynamics'] = None     # set after importing dynamics_unicycle
plant['augment'] = None      # set after importing augment_unicycle
plant['constraint'] = lambda x: abs(x[7]) > np.pi / 2 or abs(x[10]) > np.pi / 2
plant['noise'] = np.diag(np.concatenate([0.01 * np.ones(5), 0.003 * np.ones(7)]) ** 2)
plant['dt'] = dt
plant['ctrl'] = 'zoh'
plant['odei'] = odei
plant['augi'] = augi
plant['angi'] = angi
plant['poli'] = poli
plant['dyno'] = dyno
plant['dyni'] = dyni
plant['difi'] = difi
plant['propagate'] = None    # set after importing propagated

# 4. Set up the policy structure
policy = {}
policy['maxU'] = np.array([10.0, 50.0])
policy['p'] = {}
policy['p']['w'] = 1e-2 * np.random.randn(len(policy['maxU']), len(poli))
policy['p']['b'] = np.zeros(len(policy['maxU']))

# 5. Set up the cost structure
cost = {}
cost['gamma'] = 1              # discount factor
cost['p'] = np.array([0.22, 0.81])  # radius of wheel and length of rod
cost['width'] = np.array([1.0])     # cost function width
cost['expl'] = 0.0             # exploration parameter (UCB)

# 6. Set up the GP dynamics model structure
dynmodel = {}
# dynmodel['fcn'] will be set to gp1d after import
# dynmodel['train'] will be set to train after import
dynmodel['induce'] = np.zeros((300, 0, 1))  # shared inducing inputs (sparse GP)

# 7. Parameters for policy optimization
opt = {}
opt['length'] = -150            # max. number of function evaluations
opt['MFEPLS'] = 20              # max. number of function evaluations per line search
opt['verbosity'] = 2            # verbosity: 0-3
opt['method'] = 'BFGS'
trainOpt = np.array([300, 300])  # max line searches: full GP, sparse GP (FITC)

# 8. Plotting verbosity
plotting = {}
plotting['verbosity'] = 2       # 0: no plots, 1: some plots, 2: all plots

# 9. Some initializations
x = None; y = None
fantasy_mean = None; fantasy_std = None
realCost = [None] * N
M_cell = [None] * N
Sigma_cell = [None] * N
