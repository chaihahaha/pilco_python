import numpy as np
from pilco_python.base.value import value as pilco_value
from pilco_python.base.analytical_variance import compute_trajectory_and_stats
from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def value_de(p, m0, S0, dynmodel, policy, plant, cost, H,
             bo_config=None, compute_gradients=True):
    if bo_config is None or not bo_config.get('enabled', False):
        return pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                          compute_gradients=compute_gradients)

    bo_type = bo_config.get('type', 'none')
    beta = bo_config.get('beta', 1.0)
    E_remaining = bo_config.get('E_remaining', 10)
    seed = bo_config.get('seed', 42)

    _, _, _, var_c = compute_trajectory_and_stats(
        p, m0, S0, dynmodel, policy, plant, cost, H)

    mu_c = pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                       compute_gradients=False)
    if isinstance(mu_c, tuple): mu_c = mu_c[0]
    mu_c, var_c = float(mu_c), max(var_c, 1e-12)
    sigma_c = np.sqrt(var_c)

    from pilco_python.base.directed_explore import (
        ucb, gittins_index, _solve_gittins_lambda_prime)

    if bo_type == 'ucb':
        bo_val = ucb(mu_c, sigma_c, beta)
        dBO_dmu, dBO_dsigma = 1.0, -beta
    elif bo_type == 'gi':
        if E_remaining <= 1:
            bo_val, dBO_dmu, dBO_dsigma = mu_c, 1.0, 0.0
        else:
            lam = _solve_gittins_lambda_prime(1.0 - 1.0 / E_remaining)
            bo_val = mu_c + lam * sigma_c
            dBO_dmu, dBO_dsigma = 1.0, lam
    elif bo_type == 'std':
        bo_val, dBO_dmu, dBO_dsigma = mu_c - sigma_c, 1.0, -1.0
    else:
        bo_val, dBO_dmu, dBO_dsigma = mu_c, 1.0, 0.0

    if not compute_gradients:
        return float(bo_val)

    # Analytical μ gradient from PILCO
    J_raw, dJdp_raw = pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                                  compute_gradients=True)
    dJdp_flat = unwrap(dJdp_raw)

    # Gradient of σ² via central differences on analytical variance (deterministic, no noise)
    p_flat = unwrap(policy['p'])
    P = len(p_flat)
    fd_eps = 0.005
    n_dirs = bo_config.get('vg_dirs', 3)
    rng = np.random.default_rng(seed + 10000)

    dvar_flat = np.zeros(P)
    for d in range(n_dirs):
        direction = rng.standard_normal(P)
        direction /= (np.linalg.norm(direction) + 1e-12)

        p_plus_flat = p_flat + fd_eps * direction
        p_minus_flat = p_flat - fd_eps * direction
        p_plus, _ = rewrap(policy['p'].copy(), p_plus_flat)
        p_minus, _ = rewrap(policy['p'].copy(), p_minus_flat)

        _, _, _, var_plus = compute_trajectory_and_stats(
            p_plus, m0, S0, dynmodel, policy, plant, cost, H)
        _, _, _, var_minus = compute_trajectory_and_stats(
            p_minus, m0, S0, dynmodel, policy, plant, cost, H)

        dvar_flat += ((var_plus - var_minus) / (2.0 * fd_eps)) * direction

    dvar_flat /= n_dirs
    dsigma_dvar = 0.5 / sigma_c
    dsigma_flat = dsigma_dvar * dvar_flat

    dBOdp_flat = dBO_dmu * dJdp_flat + dBO_dsigma * dsigma_flat

    dBOdp, _ = rewrap(policy['p'].copy() if isinstance(policy['p'], dict) else
                      policy['p'], dBOdp_flat)
    return float(bo_val), dBOdp
