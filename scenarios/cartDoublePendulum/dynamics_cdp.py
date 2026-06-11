# dynamics_cdp.py
# *Summary:* Implements the ODE for simulating the cart-double pendulum
# dynamics.
#
#    def dynamics_cdp(t, z, f=None, compute_energy=False):
#
#
# *Input arguments:*
#
#   t     current time step (called from ODE solver)
#   z     state                                                    [6 x 1]
#   f     (optional): force value (scalar)
#   compute_energy  (optional): if True, return total mechanical energy
#
# *Output arguments:*
#
#   dz    if f is given:        state derivative wrt time
#         if compute_energy:    total mechanical energy
#
#   Note: It is assumed that the state variables are of the following order:
#         x:        [m]     position of cart
#         dx:       [m/s]   velocity of cart
#         dtheta1:  [rad/s] angular velocity of inner pendulum
#         dtheta2:  [rad/s] angular velocity of outer pendulum
#         theta1:   [rad]   angle of inner pendulum
#         theta2:   [rad]   angle of outer pendulum
#
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
# Last modified: 2013-03-05
# Translated from MATLAB to Python (1-indexed to 0-indexed).

import numpy as np


def dynamics_cdp(t, z, f=None, compute_energy=False):
    ## Code
    z = np.asarray(z, dtype=np.float64).ravel()

    # set up the system
    m1 = 0.5   # [kg]     mass of cart
    m2 = 0.5   # [kg]     mass of 1st pendulum
    m3 = 0.5   # [kg]     mass of 2nd pendulum
    l2 = 0.6   # [m]      length of 1st pendulum
    l3 = 0.6   # [m]      length of 2nd pendulum
    b  = 0.1   # [Ns/m]   coefficient of friction between cart and ground
    g  = 9.82  # [m/s^2]  acceleration of gravity

    if f is not None and not compute_energy:
        f_val = float(np.asarray(f).ravel()[0])

        A = np.array([[2*(m1+m2+m3), -(m2+2*m3)*l2*np.cos(z[4]), -m3*l3*np.cos(z[5])],
                      [-(3*m2+6*m3)*np.cos(z[4]), (2*m2+6*m3)*l2, 3*m3*l3*np.cos(z[4]-z[5])],
                      [-3*np.cos(z[5]), 3*l2*np.cos(z[4]-z[5]), 2*l3]])

        rhs = np.array([2*f_val - 2*b*z[1] - (m2+2*m3)*l2*z[2]**2*np.sin(z[4]) - m3*l3*z[3]**2*np.sin(z[5]),
                        (3*m2+6*m3)*g*np.sin(z[4]) - 3*m3*l3*z[3]**2*np.sin(z[4]-z[5]),
                        3*l2*z[2]**2*np.sin(z[4]-z[5]) + 3*g*np.sin(z[5])])

        x = np.linalg.solve(A, rhs)

        dz = np.zeros(6)
        dz[0] = z[1]
        dz[1] = x[0]
        dz[2] = x[1]
        dz[3] = x[2]
        dz[4] = z[2]
        dz[5] = z[3]

        return dz

    else:

        dz = (m1+m2+m3)*z[1]**2/2 + (m2/6+m3/2)*l2**2*z[2]**2 + m3*l3**2*z[3]**2/6 \
             - (m2/2+m3)*l2*z[1]*z[2]*np.cos(z[4]) - m3*l3*z[1]*z[3]*np.cos(z[5])/2 \
             + m3*l2*l3*z[2]*z[3]*np.cos(z[4]-z[5])/2 + (m2/2+m3)*l2*g*np.cos(z[4]) \
             + m3*l3*g*np.cos(z[5])/2

        # I2 = m2*l2**2/12  # moment of inertia around pendulum midpoint (1st link)
        # I3 = m3*l3**2/12  # moment of inertia around pendulum midpoint (2nd link)
        #
        #
        # dz = m1*z[1]**2/2 + m2/2*(z[1]**2-l2*z[1]*z[2]*np.cos(z[4])) \
        #     + m3/2*(z[1]**2 - 2*l2*z[1]*z[2]*np.cos(z[4]) - l3*z[1]*z[3]*np.cos(z[5])) \
        #     + m2*l2**2*z[2]**2/8 + I2*z[2]**2/2 \
        #     + m3/2*(l2**2*z[2]**2 + l3**2*z[3]**2/4 + l2*l3*z[2]*z[3]*np.cos(z[4]-z[5])) \
        #     + I3*z[3]**2/2 \
        #     + m2*g*l2*np.cos(z[4])/2 + m3*g*(l2*np.cos(z[4])+l3*np.cos(z[5])/2)

        return dz
