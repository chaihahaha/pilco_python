# dynamics_dp.py
# *Summary:* Implements ths ODE for simulating the double pendulum 
# dynamics, where an input torque can be applied to both links, 
# f1:torque at inner joint, f2:torque at outer joint
#
#    def dynamics_dp(t, z, f1=None, f2=None, compute_derivative=True)
#
#
# *Input arguments:*
#
#   t     current time step (called from ODE solver)
#   z     state                                                    [4 x 1]
#   f1    torque at inner joint (scalar) or full control array [f1,f2]
#   f2    torque at outer joint (scalar, optional)
#   compute_derivative if True: compute time derivatives; if False: total mechanical energy
#
# *Output arguments:*
#   
#   dz    if compute_derivative=True:      state derivative wrt time
#         if compute_derivative=False:     total mechanical energy
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


def dynamics_dp(t, z, f1=None, f2=None, compute_derivative=True):
    ## Code
    z = np.asarray(z, dtype=np.float64).ravel()

    m1 = 0.5   # [kg]     mass of 1st link
    m2 = 0.5   # [kg]     mass of 2nd link
    b1 = 0.0   # [Ns/m]   coefficient of friction (1st joint)
    b2 = 0.0   # [Ns/m]   coefficient of friction (2nd joint)
    l1 = 0.5   # [m]      length of 1st pendulum
    l2 = 0.5   # [m]      length of 2nd pendulum
    g  = 9.82  # [m/s^2]  acceleration of gravity
    I1 = m1 * l1**2 / 12  # moment of inertia around pendulum midpoint (1st link)
    I2 = m2 * l2**2 / 12  # moment of inertia around pendulum midpoint (2nd link)

    if not compute_derivative: # compute total mechanical energy
      dz = (m1 * l1**2 * z[0]**2 / 8 + I1 * z[0]**2 / 2 + m2 / 2 * (l1**2 * z[0]**2
            + l2**2 * z[1]**2 / 4 + l1 * l2 * z[0] * z[1] * np.cos(z[2] - z[3])) + I2 * z[1]**2 / 2
            + m1 * g * l1 * np.cos(z[2]) / 2 + m2 * g * (l1 * np.cos(z[2]) + l2 * np.cos(z[3]) / 2))
      return dz

    # compute time derivatives

    # Handle calling conventions:
    # 1. dynamics_dp(t, z, u_vals)  -> from simulate (u_vals is array of controls)
    # 2. dynamics_dp(t, z, f1, f2)  -> direct scalar torques
    if isinstance(f1, np.ndarray) or isinstance(f1, (list, tuple)):
        u_vals = np.asarray(f1).ravel()
        u1 = float(u_vals[0]) if len(u_vals) > 0 else 0.0
        u2 = float(u_vals[1]) if len(u_vals) > 1 else 0.0
    else:
        u1 = float(f1) if f1 is not None else 0.0
        u2 = float(f2) if f2 is not None else 0.0

    A = np.array([[l1**2 * (0.25 * m1 + m2) + I1,      0.5 * m2 * l1 * l2 * np.cos(z[2] - z[3])],
                   [0.5 * m2 * l1 * l2 * np.cos(z[2] - z[3]), l2**2 * 0.25 * m2 + I2          ]])
    b = np.array([g * l1 * np.sin(z[2]) * (0.5 * m1 + m2) - 0.5 * m2 * l1 * l2 * z[1]**2 * np.sin(z[2] - z[3])
                                                             + u1 - b1 * z[0],
                   0.5 * m2 * l2 * (l1 * z[0]**2 * np.sin(z[2] - z[3]) + g * np.sin(z[3])) + u2 - b2 * z[1]])
    x = np.linalg.solve(A, b)

    dz = np.zeros(4)
    dz[0] = x[0]
    dz[1] = x[1]
    dz[2] = z[0]
    dz[3] = z[1]

    return dz
