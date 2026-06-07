import numpy as np
from scipy.stats import norm
from scipy.special import ndtr as norm_cdf


def ucb(mu_c, sigma_c, beta=1.0):
    return mu_c - beta * sigma_c


def ucb_with_grad(mu_c, sigma_c, beta=1.0):
    val = mu_c - beta * sigma_c
    dmu = 1.0
    dsigma = -beta
    return val, dmu, dsigma


def probability_of_improvement(mu_c, var_c, best_mu, best_var):
    sigma_sq = var_c + best_var
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    z = (best_mu - mu_c) / sigma
    pi_val = norm.cdf(z)
    return -pi_val


def probability_of_improvement_with_grad(mu_c, var_c, best_mu, best_var):
    sigma_sq = var_c + best_var
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    z = (best_mu - mu_c) / sigma

    pi_val = norm.cdf(z)
    neg_pi = -pi_val

    phi_z = norm.pdf(z)
    dpi_dmu = -phi_z / sigma
    dpi_dvar = phi_z * (best_mu - mu_c) / (2.0 * sigma ** 3)

    d_neg_pi_dmu = -dpi_dmu
    d_neg_pi_dvar = -dpi_dvar

    return neg_pi, d_neg_pi_dmu, d_neg_pi_dvar * sigma_sq / max(var_c, 1e-12)


def expected_improvement(mu_c, var_c, best_mu, best_var):
    sigma_sq = var_c + best_var
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    z = (best_mu - mu_c) / sigma

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    ei_val = (mu_c - best_mu) * Phi_z + sigma * phi_z
    return ei_val


def expected_improvement_with_grad(mu_c, var_c, best_mu, best_var):
    sigma_sq = var_c + best_var
    sigma = np.sqrt(max(sigma_sq, 1e-12))
    z = (best_mu - mu_c) / sigma

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    ei_val = (mu_c - best_mu) * Phi_z + sigma * phi_z

    d_ei_dmu = Phi_z
    d_ei_dsigma = phi_z
    d_sigma_dvar = 0.5 / sigma
    d_ei_dvar = d_ei_dsigma * d_sigma_dvar

    return ei_val, d_ei_dmu, d_ei_dvar * sigma_sq / max(var_c, 1e-12)


def gittins_index(mu_c, sigma_c, sigma_y=1.0, E_remaining=10):
    if E_remaining <= 1:
        return mu_c

    gamma = 1.0 - 1.0 / E_remaining
    if gamma <= 0.0:
        return mu_c

    sigma_e = max(sigma_c, 1e-12)
    s = sigma_e ** 2 / np.sqrt(max(sigma_e ** 2 + sigma_y ** 2, 1e-12))

    lambda_prime = _solve_gittins_lambda_prime(gamma)

    gi_val = mu_c + lambda_prime * s
    return gi_val


def gittins_index_with_grad(mu_c, sigma_c, sigma_y=1.0, E_remaining=10):
    if E_remaining <= 1:
        return mu_c, 1.0, 0.0

    gamma = 1.0 - 1.0 / E_remaining
    if gamma <= 0.0:
        return mu_c, 1.0, 0.0

    sigma_e = max(sigma_c, 1e-12)
    sigma_e_sq = sigma_e ** 2
    denom = np.sqrt(max(sigma_e_sq + sigma_y ** 2, 1e-12))
    s = sigma_e_sq / denom

    lambda_prime = _solve_gittins_lambda_prime(gamma)

    gi_val = mu_c + lambda_prime * s

    d_s_d_sigma_e = (sigma_e_sq / 2.0 + sigma_y ** 2) * sigma_e / (denom ** 3) * 2.0

    d_gi_d_mu = 1.0
    d_gi_d_sigma = lambda_prime * d_s_d_sigma_e

    return gi_val, d_gi_d_mu, d_gi_d_sigma


def _solve_gittins_lambda_prime(gamma, max_iter=50, tol=1e-10):
    lambda_val = 0.0
    for i in range(max_iter):
        phi_l = norm.pdf(lambda_val)
        Phi_l = norm.cdf(lambda_val)
        g = gamma * (phi_l + lambda_val * Phi_l - lambda_val) + lambda_val
        dg = gamma * (Phi_l - 1.0) + 1.0
        if abs(g) < tol:
            break
        lambda_val = lambda_val - g / dg

    return lambda_val


def get_bo_function(bo_type='ucb', **kwargs):
    if bo_type == 'ucb':
        beta = kwargs.get('beta', 1.0)
        return lambda mu, var: ucb(mu, np.sqrt(max(var, 1e-12)), beta)
    elif bo_type == 'pi':
        best_mu = kwargs.get('best_mu', 0.0)
        best_var = kwargs.get('best_var', 0.0)
        return lambda mu, var: probability_of_improvement(mu, var, best_mu, best_var)
    elif bo_type == 'ei':
        best_mu = kwargs.get('best_mu', 0.0)
        best_var = kwargs.get('best_var', 0.0)
        return lambda mu, var: expected_improvement(mu, var, best_mu, best_var)
    elif bo_type == 'gi':
        sigma_y = kwargs.get('sigma_y', 1.0)
        E_rem = kwargs.get('E_remaining', 10)
        return lambda mu, var: gittins_index(np.sqrt(max(var, 1e-12)), sigma_y=sigma_y, E_remaining=E_rem)
    else:
        return lambda mu, var: mu
