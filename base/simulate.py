# simulate.py
# *Summary:* Simulate dynamics using a given control scheme.
#
#    function next = simulate(x0, f, plant)
#
# *Input arguments:*
#
#  x0      start state (with additional control states if required)
#  f       the control setpoint for this time step
#  plant   plant structure
#    .dt        time discretization
#    .dynamics  system function
#    .ctrl      function defining control implementation
#                  @zoh - zero-order-hold control (ZOH)
#                  @foh - first-order-hold control (FOH)
#                         with optional rise time 0 < plant.tau <= dt
#                  @lag - lagged control with time constant 0 < plant.tau
#    .delay     continuous-time delay, in range [0 dt)
#
# *Output arguments:*
#
#  next    successor state (with additional control states if required)
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modification: 2012-06-30
#
# High-Level Steps
# For each time step
# # Set up the control function
# # Simulate the dynamics (by calling ODE45)
# # Update control part of the state

import numpy as np
from scipy.integrate import solve_ivp


def _zoh(f, t, par):
    """Zero-order hold control"""
    d = par['delay']
    if d == 0:
        u = f[0]
    else:
        e = d / 100.0
        t0 = t - (d - e / 2)
        if t < d - e / 2:
            u = f[0]
        elif t < d + e / 2:
            u = (1 - t0 / e) * f[0] + t0 / e * f[1]   # prevents ODE stiffness
        else:
            u = f[1]
    return u


def _foh(f, t, par):
    """First-order hold control"""
    d = par['delay']
    tau = par['tau']
    dt = par['dt']
    if tau + d < dt:
        t0 = t - d
        if t < d:
            u = f[0]
        elif t < tau + d:
            u = (1 - t0 / tau) * f[0] + t0 / tau * f[1]
        else:
            u = f[1]
    else:
        bit = d - (dt - tau)
        if t < bit:
            u = (1 - t / bit) * f[1] + t / tau * f[0]
        elif t < d:
            u = f[0]
        else:
            t0 = t + d
            u = (1 - t0 / tau) * f[0] + t0 / tau * f[2]
    return u


def _lag(f, t, par):
    """First-order lag control"""
    d = par['delay']
    tau = par['tau']
    if d == 0:
        u = f[0] + (f[1] - f[0]) * np.exp(-t / tau)
    else:
        bit = f[1] + (f[0] - f[1]) * np.exp(-d / tau)
        if t < d:
            u = f[1] + (f[0] - f[1]) * np.exp(-t / tau)
        else:
            u = bit + (f[2] - bit) * np.exp(-t / tau)
    return u


_CTRL_FUNCTIONS = {'zoh': _zoh, 'foh': _foh, 'lag': _lag}


def simulate(x0, f, plant):
    ## Code

    x0 = np.asarray(x0).ravel()
    f = np.asarray(f).ravel()
    nU = len(f)

    dt = plant['dt']
    dynamics = plant['dynamics']
    if 'delay' in plant:
        delay = plant['delay']
    else:
        delay = 0.0
    if 'tau' in plant:
        tau = plant['tau']
    else:
        tau = dt

    par = {'dt': dt, 'delay': delay, 'tau': tau}

    # 1. Set up control function ------------------------------------------------
    # f{t} = control setpoint over time t to d+dt (determined by policy)
    # u{t} = control currently being applied at time t
    con_name = plant['ctrl']
    if con_name == 'zoh' and delay == 0:                # U = [f{t}]
        x0s = x0.copy()
        U = f.reshape(1, -1)  # shape (1, nU)
        id_val = 0
    elif con_name == 'zoh' or \
         (con_name == 'foh' and tau + delay <= dt) or \
         (con_name == 'lag' and delay == 0):
        # U = [u{t} f{t}]
        x0s = x0[:-nU].copy()
        U = np.column_stack([x0[-nU:], f])  # shape (nU, 2)
        id_val = 1
    else:                                              # U = [f{t-1} u{t} f{t}]
        x0s = x0[:-2 * nU].copy()
        U = np.column_stack([x0[-2 * nU:].reshape(nU, 2), f])  # shape (nU, 3)
        id_val = 2

    ctrlfcn = _CTRL_FUNCTIONS[con_name]

    if nU > 0:
        def ode_rhs(t, y):
            """ODE right-hand side with control signals"""
            u_vals = np.array([ctrlfcn(U[j, :], t, par) for j in range(nU)])
            return dynamics(t, y, u_vals)
    else:
        def ode_rhs(t, y):
            return dynamics(t, y)

    # 2. Simulate dynamics ------------------------------------------------------
    t_eval = [0.0, dt / 2.0, dt]
    sol = solve_ivp(ode_rhs, [0.0, dt], x0s, method='RK45',
                    t_eval=t_eval, rtol=1e-12, atol=1e-12)
    x1 = sol.y[:, -1]                                                 # next state

    # 3. Update control part of the state ---------------------------------------
    udt = np.array([ctrlfcn(U[j, :], dt, par) for j in range(nU)])
    if id_val == 0:
        next_state = x1.copy()                         # return augmented state
    elif id_val == 1:
        next_state = np.concatenate([x1, udt])
    else:
        next_state = np.concatenate([x1, f, udt])

    return next_state
