# value.py
# *Summary:* Compute expected (discounted) cumulative cost for a given (set of) initial
# state distributions
#
#     function [J, dJdp] = value(p, m0, S0, dynmodel, policy, plant, cost, H)
#
# *Input arguments:*
#
#   p            policy parameters chosen by minimize
#   policy       policy structure
#     .fcn       function which implements the policy
#     .p         parameters passed to the policy
#   m0           matrix (D by k) of initial state means
#   S0           covariance matrix (D by D) for initial state
#   dynmodel     dynamics model structure
#   plant        plant structure
#   cost         cost function structure
#     .fcn       function handle to the cost
#     .gamma     discount factor
#  H             length of prediction horizon
#
# *Output arguments:*
#
#  J             expected cumulative (discounted) cost
#  dJdp          (optional) derivative of J wrt the policy parameters
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modification: 2013-03-21
#
# High-Level Steps
# # Compute distribution of next state
# # Compute corresponding expected immediate cost (discounted)
# # At end of prediction horizon: sum all immediate costs up

import numpy as np
from ..util.unwrap import unwrap
from ..util.rewrap import rewrap


def value(p, m0, S0, dynmodel, policy, plant, cost, H, compute_gradients=True, return_var_grad=False):
    """
    Compute expected (discounted) cumulative cost for a given initial state distribution.

    Parameters
    ----------
    p : ndarray
        Policy parameters (structured, as rewrapped by minimize)
    m0 : ndarray (D,) or (D, 1)
        Initial state mean
    S0 : ndarray (D, D)
        Initial state covariance
    dynmodel : dict
        Dynamics model structure
    policy : dict
        Policy structure with 'p' and 'fcn' fields
    plant : dict
        Plant structure with 'prop' function
    cost : dict
        Cost structure with 'fcn' function and 'gamma' discount factor
    H : int
        Length of prediction horizon
    compute_gradients : bool, optional
        If True, compute dJdp. Default True.

    Returns
    -------
    J : float
        Expected cumulative (discounted) cost
    dJdp : ndarray
        Derivative of J wrt policy parameters (same structure as policy['p'])
    """
    # Code

    policy['p'] = p            # overwrite policy.p with new parameters from minimize
    p_flat = unwrap(policy['p'])
    dp_flat = np.zeros_like(p_flat)

    m = m0.copy()
    S = S0.copy()
    L = np.zeros(H)

    if not compute_gradients:                                       # no derivatives required

        for t in range(H):                                  # for all time steps in horizon
            result = plant['prop'](m, S, plant, dynmodel, policy)      # get next state
            m = result[0]
            S = result[1]
            L[t] = (cost['gamma'] ** (t + 1)) * cost['fcn'](cost, m, S)[0]  # expected discounted cost

    else:                                               # otherwise, get derivatives

        dmOdp = np.zeros((m0.shape[0], len(p_flat)))
        dSOdp = np.zeros((m0.shape[0] * m0.shape[0], len(p_flat)))
        dS2_flat = np.zeros_like(p_flat)

        for t in range(H):                                  # for all time steps in horizon
            result = plant['prop'](m, S, plant, dynmodel, policy, compute_derivatives=True)  # get next state
            m = result[0]
            S = result[1]
            dmdmO = result[2]
            dSdmO = result[3]
            dmdSO = result[4]
            dSdSO = result[5]
            dmdp = result[6]
            dSdp = result[7]

            dmdp_total = dmdmO @ dmOdp + dmdSO @ dSOdp + dmdp
            dSdp_total = dSdmO @ dmOdp + dSdSO @ dSOdp + dSdp

            res_cost = cost['fcn'](cost, m, S)                       # predictive cost
            L_t = res_cost[0]
            dLdm = res_cost[1]
            dLdS = res_cost[2]
            s2dM = res_cost[4] if len(res_cost) > 4 else np.zeros_like(dLdm)
            s2dS = res_cost[5] if len(res_cost) > 5 else np.zeros_like(dLdS)
            L[t] = ((cost['gamma'] ** (t + 1)) * L_t).item()

            dLdm_flat = dLdm.ravel()
            dLdS_flat = dLdS.ravel()
            dp_flat = dp_flat + (cost['gamma'] ** (t + 1)) * \
                (dLdm_flat @ dmdp_total + dLdS_flat @ dSdp_total)

            s2dM_flat = s2dM.ravel()
            s2dS_flat = s2dS.ravel()
            dS2_flat = dS2_flat + (cost['gamma'] ** (t + 1)) * \
                (s2dM_flat @ dmdp_total + s2dS_flat @ dSdp_total)

            dmOdp = dmdp_total.copy()
            dSOdp = dSdp_total.copy()                                 # bookkeeping

    J = np.sum(L)
    if compute_gradients:
        dJdp, _ = rewrap(policy['p'].copy() if isinstance(policy['p'], dict) else policy['p'], dp_flat)
        if return_var_grad:
            dS2dp, _ = rewrap(policy['p'].copy() if isinstance(policy['p'], dict) else policy['p'], dS2_flat)
            return J, dJdp, dS2dp
        return J, dJdp
    else:
        return J
