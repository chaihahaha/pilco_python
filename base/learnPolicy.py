import numpy as np
from ..util.minimize import minimize


def learn_policy(mu0Sim, S0Sim, dynmodel, policy, plant, cost, H, plotting=None,
                 value_func=None, pred_func=None, calc_cost_func=None):
    """Policy search using MATLAB-compatible BFGS minimize. Matches learnPolicy.m."""
    if value_func is None:
        from .value import value as value_func
    if pred_func is None:
        from .pred import pred as pred_func
    if calc_cost_func is None:
        from .calcCost import calcCost as calc_cost_func

    opt = {'length': 150, 'MFEPLS': 30, 'verbosity': 1}
    policy['p'], fX3, _, _ = minimize(
        policy['p'], value_func, opt, mu0Sim, S0Sim,
        dynmodel, policy, plant, cost, H)

    if plotting is not None and 'verbosity' in plotting and plotting['verbosity'] > 1:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.plot(fX3)
        plt.xlabel('line search iteration')
        plt.ylabel('function value')
        plt.draw()

    M, Sigma = pred_func(policy, plant, dynmodel, mu0Sim, S0Sim, H)
    fantasy_mean, fantasy_std = calc_cost_func(cost, M, Sigma)

    if plotting is not None and 'verbosity' in plotting and plotting['verbosity'] > 0:
        import matplotlib.pyplot as plt
        plt.figure(3)
        plt.clf()
        plt.errorbar(range(H + 1), fantasy_mean, 2 * fantasy_std)
        plt.xlabel('time step')
        plt.ylabel('immediate cost')
        plt.draw()

    return policy, fX3, M, Sigma, fantasy_mean, fantasy_std
