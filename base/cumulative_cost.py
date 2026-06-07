import numpy as np
from pilco_python.base.propagate import propagate


def _propagate_cost(cost, m, s, plant, dynmodel, policy, H):
    total_cost = 0.0
    for t in range(H):
        m, s = plant['prop'](m, s, plant, dynmodel, policy)
        L_t = cost['fcn'](cost, m, s)[0]
        total_cost += L_t
    return total_cost


def mc_cumulative_cost_stats(policy, plant, dynmodel, cost, m0, S0, H,
                              n_samples=500, seed=42):
    rng = np.random.default_rng(seed)
    D0 = len(m0)

    L_chol = np.linalg.cholesky(S0)
    costs = np.zeros(n_samples)

    for k in range(n_samples):
        z = rng.standard_normal(D0)
        m_k = m0 + L_chol @ z

        costs[k] = _propagate_cost(
            cost, m_k, np.zeros((D0, D0)), plant, dynmodel, policy, H)

    mu_c = np.mean(costs)
    var_c = np.var(costs, ddof=1)

    return mu_c, max(var_c, 1e-12)


def distributional_cost_stats(policy, plant, dynmodel, cost, m0, S0, H,
                               n_samples=300, seed=42):
    rng = np.random.default_rng(seed)
    D0 = len(m0)

    L_chol = np.linalg.cholesky(S0)
    costs = np.zeros(n_samples)

    for k in range(n_samples):
        z = rng.standard_normal(D0)
        m_k = m0 + L_chol @ z
        m = m_k.copy()
        s = np.zeros((D0, D0))

        total = 0.0
        for t in range(H):
            m_next, s_next = propagate(m, s, plant, dynmodel, policy)
            total += cost['fcn'](cost, m_next, s_next)[0]
            m = m_next
            s = s_next

        costs[k] = total

    mu_c = np.mean(costs)
    var_c = np.var(costs, ddof=1)

    return mu_c, max(var_c, 1e-12), costs
