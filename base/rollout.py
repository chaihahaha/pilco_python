# rollout.py
# *Summary:* Generate a state trajectory using an ODE solver (and any additional
# dynamics) from a particular initial state by applying either a particular
# policy or random actions.
#
#   function [x y L latent] = rollout(start, policy, H, plant, cost)
#
# *Input arguments:*
#
#   start       vector containing initial states (without controls)   [nX  x  1]
#   policy      policy structure
#     .fcn        policy function
#     .p          parameter structure (if empty: use random actions)
#     .maxU       vector of control input saturation values           [nU  x  1]
#   H           rollout horizon in steps
#   plant       the dynamical system structure
#     .subplant   (opt) additional discrete-time dynamics
#     .augment    (opt) augment state using a known mapping
#     .constraint (opt) stop rollout if violated
#     .poli       indices for states passed to the policy
#     .dyno       indices for states passed to cost
#     .odei       indices for states passed to the ode solver
#     .subi       (opt) indices for states passed to subplant function
#     .augi       (opt) indices for states passed to augment function
#   cost    cost structure
#
# *Output arguments:*
#
#   x          matrix of observed states                           [H   x nX+nU]
#   y          matrix of corresponding observed successor states   [H   x   nX ]
#   L          cost incurred at each time step                     [ 1  x    H ]
#   latent     matrix of latent states                             [H+1 x   nX ]
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modification: 2013-05-21
#
# High-Level Steps
#
# # Compute control signal $u$ from state $x$:
# either apply policy or random actions
# # Simulate the true dynamics for one time step using the current pair $(x,u)$
# # Check whether any constraints are violated (stop if true)
# # Apply random noise to the successor state
# # Compute cost (optional)
# # Repeat until end of horizon


import numpy as np
from pilco_python.util.gTrig import gTrig
from .simulate import simulate


def rollout(start, policy, H, plant, cost=None, compute_cost=False):
    ## Code

    start = np.asarray(start).ravel()

    augment_fn = plant.get('augment', lambda x: np.array([]))
    augi = plant.get('augi', np.array([], dtype=int))
    subplant_fn = plant.get('subplant', lambda state, u: np.array([]))
    subi = plant.get('subi', np.array([], dtype=int))

    odei = plant['odei']
    poli = plant['poli']
    dyno = plant['dyno']
    angi = plant['angi']
    simi = np.sort(np.concatenate([odei, subi]))
    nX = len(simi) + len(augi)
    nU = len(policy['maxU'])
    nA = len(angi)

    state = np.zeros(len(simi) + len(augi)); state[simi] = start
    state[augi] = augment_fn(state)          # initializations
    noise = plant['noise']
    if np.isscalar(noise) or noise.size == 1:
        noise_chol = np.sqrt(np.atleast_1d(noise).ravel())
    else:
        noise_chol = np.linalg.cholesky(noise)

    x = np.zeros((H + 1, nX + 2 * nA))
    if np.isscalar(noise) or noise.size == 1:
        x[0, simi] = start + np.random.randn(len(simi)) * noise_chol
    else:
        x[0, simi] = start + np.random.randn(len(simi)) @ noise_chol.T
    x[0, augi] = augment_fn(x[0, :])
    u = np.zeros((H, nU))
    latent = np.zeros((H + 1, len(state) + nU))
    y = np.zeros((H, nX))
    L = np.zeros(H)
    next_state = np.zeros(len(simi))

    for i_idx in range(H):  # --------------------------------------------- generate trajectory
        s = x[i_idx, dyno]
        sa = gTrig(s, np.zeros((len(s), len(s))), angi, compute_derivatives=False)[0]
        s_aug = np.concatenate([s, sa])
        x[i_idx, -2 * nA:] = s_aug[-2 * nA:]

        # 1. Apply policy ... or random actions --------------------------------------
        if policy.get('fcn') is not None:
            poli_input = s_aug[poli].ravel() if len(poli) > 0 else s_aug
            u_val = policy['fcn'](policy, poli_input, np.zeros((len(poli), len(poli))),
                                  compute_derivatives=False)
            if isinstance(u_val, tuple):
                u[i_idx, :] = np.atleast_1d(np.asarray(u_val[0]).ravel())
            else:
                u[i_idx, :] = np.atleast_1d(np.asarray(u_val).ravel())
        else:
            u[i_idx, :] = policy['maxU'] * (2 * np.random.rand(nU) - 1)

        latent[i_idx, :] = np.concatenate([state, u[i_idx, :]])          # latent state

        # 2. Simulate dynamics -------------------------------------------------------
        next_state_odei = simulate(state[odei], u[i_idx, :].ravel(), plant)
        next_state[odei] = next_state_odei[:len(odei)]  # take state part
        next_state[subi] = subplant_fn(state, u[i_idx, :])

        # 3. Stop rollout if constraints violated ------------------------------------
        if 'constraint' in plant and plant['constraint'](next_state[odei]):
            H = i_idx  # truncate horizon
            print('state constraints violated...')
            break

        # 4. Augment state and randomize ---------------------------------------------
        state[simi] = next_state[simi]
        state[augi] = augment_fn(state)
        if np.isscalar(noise) or noise.size == 1:
            x[i_idx + 1, simi] = state[simi] + np.random.randn(len(simi)) * noise_chol
        else:
            x[i_idx + 1, simi] = state[simi] + np.random.randn(len(simi)) @ noise_chol.T
        x[i_idx + 1, augi] = augment_fn(x[i_idx + 1, :])

        # 5. Compute Cost ------------------------------------------------------------
        if compute_cost and cost is not None:
            L_val = cost['fcn'](cost, state[dyno], np.zeros((len(dyno), len(dyno))))
            L[i_idx] = L_val[0] if isinstance(L_val, tuple) else L_val

    y = x[1:H + 1, :nX]
    x_out = np.column_stack([x[:H, :], u[:H, :]])
    latent[H, :nX] = state
    latent = latent[:H + 1, :]
    L = L[:H]

    return x_out, y, L, latent
