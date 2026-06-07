import numpy as np
from pilco_python.base.value import value as pilco_value
from pilco_python.base.cumulative_cost import distributional_cost_stats
from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def value_de(p, m0, S0, dynmodel, policy, plant, cost, H,
             bo_config=None, compute_gradients=True):

    J_raw, dJdp_raw = pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                                  compute_gradients=True)

    if bo_config is None or not bo_config.get('enabled', False):
        if compute_gradients:
            return float(J_raw), dJdp_raw
        return float(J_raw)

    bo_type = bo_config.get('type', 'none')
    n_mc = bo_config.get('n_mc', 100)
    seed = bo_config.get('seed', 42)
    beta = bo_config.get('beta', 1.0)
    sigma_y = bo_config.get('sigma_y', 1.0)
    E_remaining = bo_config.get('E_remaining', 10)

    mu_c, var_c, costs = distributional_cost_stats(
        policy, plant, dynmodel, cost, m0, S0, H,
        n_samples=n_mc, seed=seed)
    sigma_c = np.sqrt(var_c)

    from pilco_python.base.directed_explore import (
        ucb, probability_of_improvement, expected_improvement,
        gittins_index
    )

    if bo_type == 'ucb':
        bo_val = ucb(mu_c, sigma_c, beta)
    elif bo_type == 'gi':
        bo_val = gittins_index(mu_c, sigma_c, sigma_y=sigma_y,
                               E_remaining=E_remaining)
    elif bo_type == 'std':
        bo_val = mu_c - sigma_c
    else:
        bo_val = float(mu_c)

    if not compute_gradients:
        return float(bo_val)

    dJdp_flat = unwrap(dJdp_raw)
    dBOdp_flat = 1.0 * dJdp_flat
    dBOdp, _ = rewrap(dJdp_raw.copy() if isinstance(dJdp_raw, dict) else
                     policy['p'].copy(), dBOdp_flat)

    return float(bo_val), dBOdp
