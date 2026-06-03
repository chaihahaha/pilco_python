# dynamics_pendulum.py
# *Summary:* Implements the ODE for simulating the pendulum dynamics, where
# an input torque f can be applied
#
#    def dynamics_pendulum(t,z,u)
#
#
# *Input arguments:*
#
#		t     current time step (called from ODE solver)
#   z     state                                                    [2 x 1]
#   u     (optional): torque f(t) applied to pendulum
#
# *Output arguments:*
#
#   dz    state derivative wrt time
#
#   Note: It is assumed that the state variables are of the following order:
#         z[0]:  [rad/s] angular velocity of pendulum
#         z[1]:  [rad]   angle of pendulum
#
# A detailed derivation of the dynamics can be found in:
#
# M.P. Deisenroth:
# Efficient Reinforcement Learning Using Gaussian Processes, Appendix C,
# KIT Scientific Publishing, 2010.
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-18

import numpy as np


def dynamics_pendulum(t, z, u):
    ## Code

    l = 1.0     # [m]        length of pendulum
    m = 1.0     # [kg]       mass of pendulum
    g = 9.82    # [m/s^2]    acceleration of gravity
    b = 0.01    # [s*Nm/rad] friction coefficient

    u_val = np.asarray(u).ravel()[0]  # u(t) in MATLAB
    z_arr = np.asarray(z, dtype=np.float64).ravel()

    dz = np.zeros(2)
    dz[0] = (u_val - b * z_arr[0] - m * g * l * np.sin(z_arr[1]) / 2) / (m * l**2 / 3)
    dz[1] = z_arr[0]

    return dz
