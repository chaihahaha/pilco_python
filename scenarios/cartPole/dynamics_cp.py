# dynamics_cp.py
# *Summary:* Implements the ODE for simulating the cart-pole dynamics.
#
#    def dynamics_cp(t, z, f=None)
#
# *Input arguments:*
#
#   t     current time step (called from ODE solver)
#   z     state                                                    [4 x 1]
#   f     (optional): force f(t), callable
#
# *Output arguments:*
#
#   dz    if f is provided:      state derivative wrt time
#         if f is None:          total mechanical energy
#
# Note: It is assumed that the state variables are of the following order:
#       x:        [m]     position of cart
#       dx:       [m/s]   velocity of cart
#       dtheta:   [rad/s] angular velocity
#       theta:    [rad]   angle
#
# A detailed derivation of the dynamics can be found in:
#
# M.P. Deisenroth:
# Efficient Reinforcement Learning Using Gaussian Processes, Appendix C,
# KIT Scientific Publishing, 2010.
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-03-08

import numpy as np


def dynamics_cp(t, z, f=None):
    l = 0.5   # [m]      length of pendulum
    m = 0.5   # [kg]     mass of pendulum
    M = 0.5   # [kg]     mass of cart
    b = 0.1   # [N/m/s]  coefficient of friction between cart and ground
    g = 9.82  # [m/s^2]  acceleration of gravity

    z = np.asarray(z, dtype=np.float64).ravel()

    if f is not None:
        dz = np.zeros(4)
        f_val = float(np.asarray(f).ravel()[0]) if callable(f) else float(np.asarray(f).ravel()[0]) if not np.isscalar(f) else float(f)
        dz[0] = z[1]
        dz[1] = (2 * m * l * z[2]**2 * np.sin(z[3]) + 3 * m * g * np.sin(z[3]) * np.cos(z[3])
                 + 4 * f_val - 4 * b * z[1]) / (4 * (M + m) - 3 * m * np.cos(z[3])**2)
        dz[2] = (-3 * m * l * z[2]**2 * np.sin(z[3]) * np.cos(z[3]) - 6 * (M + m) * g * np.sin(z[3])
                 - 6 * (f_val - b * z[1]) * np.cos(z[3])) / (4 * l * (m + M) - 3 * m * l * np.cos(z[3])**2)
        dz[3] = z[2]
        return dz
    else:
        dz = (M + m) * z[1]**2 / 2 + 1/6 * m * l**2 * z[2]**2 + m * l * (z[1] * z[2] - g) * np.cos(z[3]) / 2
        return dz
