import numpy as np
from pilco_python.base.value import value as pilco_value
from pilco_python.base.analytical_variance import compute_trajectory_and_stats
from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def _finite_diff_var_gradient(policy, plant, dynmodel, cost, m0, S0, H,
                               fd_eps=0.005, n_dirs=4, seed=0):
    p_flat = unwrap(policy['p'])
    P = len(p_flat)

    p_dict, _ = rewrap(policy['p'].copy(), p_flat)
    _, _, _, var_center, _ = compute_trajectory_and_stats(
        p_dict, m0, S0, dynmodel, policy, plant, cost, H)

    grad_var = np.zeros(P)
    rng = np.random.default_rng(seed + 1)

    for d in range(n_dirs):
        direction = rng.standard_normal(P)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        pp_flat = p_flat + fd_eps * direction
        pm_flat = p_flat - fd_eps * direction

        p_plus, _ = rewrap(policy['p'].copy(), pp_flat)
        p_minus, _ = rewrap(policy['p'].copy(), pm_flat)

        _, _, _, var_plus, _ = compute_trajectory_and_stats(
            p_plus, m0, S0, dynmodel, policy, plant, cost, H)
        _, _, _, var_minus, _ = compute_trajectory_and_stats(
            p_minus, m0, S0, dynmodel, policy, plant, cost, H)

        dir_deriv = (var_plus - var_minus) / (2.0 * fd_eps)
        grad_var += dir_deriv * direction

    return grad_var.reshape(-1, 1) / n_dirs, var_center


def value_de(p, m0, S0, dynmodel, policy, plant, cost, H,
             bo_config=None, compute_gradients=True):

    if bo_config is None or not bo_config.get('enabled', False):
        return pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                          compute_gradients=compute_gradients)

    bo_type = bo_config.get('type', 'none')
    seed = bo_config.get('seed', 42)
    beta = bo_config.get('beta', 1.0)
    sigma_y = bo_config.get('sigma_y', 0.5)
    E_remaining = bo_config.get('E_remaining', 10)

    _, _, _, var_c, _ = compute_trajectory_and_stats(
        p, m0, S0, dynmodel, policy, plant, cost, H)
    mu_c = pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                      compute_gradients=False)
    if isinstance(mu_c, tuple):
        mu_c = mu_c[0]
    mu_c = float(mu_c)
    sigma_c = np.sqrt(max(var_c, 1e-12))

    from pilco_python.base.directed_explore import (
        ucb, gittins_index, _solve_gittins_lambda_prime
    )

    if bo_type == 'ucb':
        bo_val = ucb(mu_c, sigma_c, beta)
        dBO_dmu, dBO_dsigma = 1.0, -beta
    elif bo_type == 'gi':
        if E_remaining <= 1:
            bo_val = mu_c
            dBO_dmu, dBO_dvar = 1.0, 0.0
        else:
            gamma = 1.0 - 1.0 / E_remaining
            lam = _solve_gittins_lambda_prime(gamma)
            s = var_c / np.sqrt(max(var_c + sigma_y ** 2, 1e-12))
            bo_val = mu_c + lam * s
            dBO_dmu = 1.0
            ds_dvar = (var_c / 2.0 + sigma_y ** 2) / (
                max(var_c + sigma_y ** 2, 1e-12) ** 1.5)
            dBO_dvar = lam * ds_dvar
    elif bo_type == 'std':
        bo_val = mu_c - sigma_c
        dBO_dmu, dBO_dsigma = 1.0, -1.0
    else:
        bo_val = mu_c
        dBO_dmu, dBO_sigma = 1.0, 0.0

    if not compute_gradients:
        return float(bo_val)

    J_raw, dJdp_raw = pilco_value(p, m0, S0, dynmodel, policy, plant, cost, H,
                                  compute_gradients=True)
    dJdp_flat = unwrap(dJdp_raw)

    use_var_grad = bo_config.get('use_var_grad', False)
    if use_var_grad:
        vg_dirs = bo_config.get('vg_dirs', 3)
        dvar_dp, _ = _finite_diff_var_gradient(
            policy, plant, dynmodel, cost, m0, S0, H,
            fd_eps=0.005, n_dirs=vg_dirs, seed=seed + 10000)
        dvar_dp = dvar_dp.ravel()

    if use_var_grad and bo_type == 'gi':
        dBOdp_flat = dBO_dmu * dJdp_flat + dBO_dvar * dvar_dp
    elif use_var_grad and bo_type in ('ucb', 'std'):
        dsigma_dvar = 0.5 / max(sigma_c, 1e-12)
        dsigma_dp = dsigma_dvar * dvar_dp
        dBOdp_flat = dBO_dmu * dJdp_flat + dBO_dsigma * dsigma_dp
    else:
        dBOdp_flat = dBO_dmu * dJdp_flat

    dBOdp, _ = rewrap(
        policy['p'].copy() if isinstance(policy['p'], dict) else policy['p'],
        dBOdp_flat)
    return float(bo_val), dBOdp
