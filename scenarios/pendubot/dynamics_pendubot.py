# dynamics_pendubot.py
# *Summary:* Implements the ODE for simulating the Pendubot
# dynamics, where an input torque f can be applied to the inner link
#
#    def dynamics_pendubot(t, z, f, compute_energy=False)
#
#
# *Input arguments:*
#
#   t     current time step (called from ODE solver)
#   z     state                                                    [4 x 1]
#   f     (optional): torque f(t) applied to inner pendulum
#   compute_energy  (flag): if True, return total mechanical energy
#                           instead of state derivatives
#
# *Output arguments:*
#
#   dz    if compute_energy=False:   state derivative wrt time
#         if compute_energy=True:    total mechanical energy
#
#   Note: It is assumed that the state variables are of the following order:
#         dtheta1:  [rad/s] angular velocity of inner pendulum
#         dtheta2:  [rad/s] angular velocity of outer pendulum
#         theta1:   [rad]   angle of inner pendulum
#         theta2:   [rad]   angle of outer pendulum
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
# Last modified: 2013-03-08

import numpy as np


def dynamics_pendubot(t, z, f=None, compute_energy=False):
    ## Code
    z = np.asarray(z, dtype=np.float64).ravel()
    m1 = 0.5   # [kg]     mass of 1st link
    m2 = 0.5   # [kg]     mass of 2nd link
    b1 = 0.0   # [Ns/m]  coefficient of friction (1st joint)
    b2 = 0.0   # [Ns/m]  coefficient of friction (2nd joint)
    l1 = 0.5   # [m]      length of 1st pendulum
    l2 = 0.5   # [m]      length of 2nd pendulum
    g  = 9.82  # [m/s^2]  acceleration of gravity
    I1 = m1 * l1**2 / 12   # moment of inertia around pendulum midpoint (inner link)
    I2 = m2 * l2**2 / 12   # moment of inertia around pendulum midpoint (outer link)

    if not compute_energy:             # compute time derivatives
        if f is None:
            f_val = 0.0
        elif callable(f):
            f_val = f(t)
        else:
            f_val = float(np.asarray(f).ravel()[0])

        A = np.array([[l1**2 * (0.25 * m1 + m2) + I1,
                       0.5 * m2 * l1 * l2 * np.cos(z[2] - z[3])],
                      [0.5 * m2 * l1 * l2 * np.cos(z[2] - z[3]),
                       l2**2 * 0.25 * m2 + I2]])
        b = np.array([
            g * l1 * np.sin(z[2]) * (0.5 * m1 + m2) -
            0.5 * m2 * l1 * l2 * z[1]**2 * np.sin(z[2] - z[3]) + f_val - b1 * z[0],
            0.5 * m2 * l2 * (l1 * z[0]**2 * np.sin(z[2] - z[3]) +
                             g * np.sin(z[3])) - b2 * z[1]
        ])
        x = np.linalg.solve(A, b)

        dz = np.zeros(4)
        dz[0] = x[0]
        dz[1] = x[1]
        dz[2] = z[0]
        dz[3] = z[1]
        return dz

    else:                               # compute total mechanical energy
        dz = (m1 * l1**2 * z[0]**2 / 8 +
              I1 * z[0]**2 / 2 +
              m2 / 2 * (l1**2 * z[0]**2 +
                        l2**2 * z[1]**2 / 4 +
                        l1 * l2 * z[0] * z[1] * np.cos(z[2] - z[3])) +
              I2 * z[1]**2 / 2 +
              m1 * g * l1 * np.cos(z[2]) / 2 +
              m2 * g * (l1 * np.cos(z[2]) + l2 * np.cos(z[3]) / 2))
        return dz
