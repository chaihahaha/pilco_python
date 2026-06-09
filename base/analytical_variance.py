import numpy as np
from pilco_python.util.gTrig import gTrig


def augment_state(mu, S, angi):
    D0 = len(mu)
    D1 = D0 + 2 * len(angi)

    mu_aug = np.zeros(D1)
    mu_aug[:D0] = mu.ravel()

    if len(angi) == 0:
        S_aug = np.zeros((D1, D1))
        S_aug[:D0, :D0] = S
        return mu_aug, S_aug

    m_gtrig, s_gtrig, C = gTrig(mu.ravel(), np.atleast_2d(S), angi,
                                 compute_derivatives=False)
    mu_aug[D0:] = m_gtrig.ravel()
    S_aug = np.zeros((D1, D1))
    S_aug[:D0, :D0] = S
    S_aug[D0:, D0:] = s_gtrig
    cross = S @ C
    S_aug[:D0, D0:] = cross
    S_aug[D0:, :D0] = cross.T
    return mu_aug, S_aug


def augment_cross_covariance(mu1, mu2, S11, S22, S12, angi):
    D0 = len(mu1)
    D1 = D0 + 2 * len(angi)
    mu_a1, mu_a2 = np.zeros(D1), np.zeros(D1)
    mu_a1[:D0], mu_a2[:D0] = mu1.ravel(), mu2.ravel()
    S_a11, S_a22, S_a12 = np.zeros((D1, D1)), np.zeros((D1, D1)), np.zeros((D1, D1))
    S_a11[:D0, :D0], S_a22[:D0, :D0], S_a12[:D0, :D0] = S11, S22, S12

    for ai, ang_idx in enumerate(angi):
        aa = D0 + 2 * ai
        m_i, m_j = mu1[ang_idx], mu2[ang_idx]
        v_i = S11[ang_idx, ang_idx]
        v_j = S22[ang_idx, ang_idx]
        v_ij = S12[ang_idx, ang_idx]

        ci, cj = np.exp(-v_i / 2), np.exp(-v_j / 2)

        mu_a1[aa] = np.sin(m_i) * ci
        mu_a1[aa + 1] = np.cos(m_i) * ci
        mu_a2[aa] = np.sin(m_j) * cj
        mu_a2[aa + 1] = np.cos(m_j) * cj

        diff_m, sum_m = m_i - m_j, m_i + m_j
        v_diff = v_i + v_j - 2 * v_ij
        v_sum = v_i + v_j + 2 * v_ij
        ed = np.exp(-v_diff / 2)
        es = np.exp(-v_sum / 2)

        E_ss = 0.5 * (np.cos(diff_m) * ed - np.cos(sum_m) * es)
        E_cc = 0.5 * (np.cos(diff_m) * ed + np.cos(sum_m) * es)
        E_sc = 0.5 * (np.sin(sum_m) * es + np.sin(diff_m) * ed)
        E_cs = 0.5 * (np.sin(sum_m) * es - np.sin(diff_m) * ed)

        S_a11[aa, aa] = 0.5 * (1 - np.cos(2 * m_i) * np.exp(-2 * v_i)) - mu_a1[aa] ** 2
        S_a11[aa, aa + 1] = 0.5 * np.sin(2 * m_i) * np.exp(-2 * v_i) - mu_a1[aa] * mu_a1[aa + 1]
        S_a11[aa + 1, aa] = S_a11[aa, aa + 1]
        S_a11[aa + 1, aa + 1] = 0.5 * (1 + np.cos(2 * m_i) * np.exp(-2 * v_i)) - mu_a1[aa + 1] ** 2

        S_a22[aa, aa] = 0.5 * (1 - np.cos(2 * m_j) * np.exp(-2 * v_j)) - mu_a2[aa] ** 2
        S_a22[aa, aa + 1] = 0.5 * np.sin(2 * m_j) * np.exp(-2 * v_j) - mu_a2[aa] * mu_a2[aa + 1]
        S_a22[aa + 1, aa] = S_a22[aa, aa + 1]
        S_a22[aa + 1, aa + 1] = 0.5 * (1 + np.cos(2 * m_j) * np.exp(-2 * v_j)) - mu_a2[aa + 1] ** 2

        S_a12[aa, aa] = E_ss - mu_a1[aa] * mu_a2[aa]
        S_a12[aa + 1, aa] = E_cs - mu_a1[aa + 1] * mu_a2[aa]
        S_a12[aa, aa + 1] = E_sc - mu_a1[aa] * mu_a2[aa + 1]
        S_a12[aa + 1, aa + 1] = E_cc - mu_a1[aa + 1] * mu_a2[aa + 1]

        cos_i, neg_s_i = np.cos(m_i) * ci, -np.sin(m_i) * ci
        cos_j, neg_s_j = np.cos(m_j) * cj, -np.sin(m_j) * cj

        for b in range(D0):
            if b == ang_idx:
                continue
            cv_i = S12[b, ang_idx]
            S_a12[b, aa] = cv_i * cos_j
            S_a12[b, aa + 1] = cv_i * neg_s_j
            S_a12[aa, b] = cv_i * cos_i
            S_a12[aa + 1, b] = cv_i * neg_s_i

        S_a11[aa, ang_idx] = v_i * cos_i - mu_a1[aa] * mu_a1[ang_idx]
        S_a11[aa + 1, ang_idx] = v_i * neg_s_i - mu_a1[aa + 1] * mu_a1[ang_idx]
        S_a11[ang_idx, aa] = S_a11[aa, ang_idx]
        S_a11[ang_idx, aa + 1] = S_a11[aa + 1, ang_idx]

        S_a22[aa, ang_idx] = v_j * cos_j - mu_a2[aa] * mu_a2[ang_idx]
        S_a22[aa + 1, ang_idx] = v_j * neg_s_j - mu_a2[aa + 1] * mu_a2[ang_idx]
        S_a22[ang_idx, aa] = S_a22[aa, ang_idx]
        S_a22[ang_idx, aa + 1] = S_a22[aa + 1, ang_idx]

    return mu_a1, mu_a2, S_a11, S_a22, S_a12


def expected_sat_cost(mu, Sigma, z, W):
    D = len(mu)
    In = np.eye(D)
    m_v = mu.reshape(-1, 1)
    z_v = z.reshape(-1, 1)

    iSpW = np.linalg.solve((In + Sigma @ W).T, W.T).T
    det_val = np.linalg.det(In + Sigma @ W)
    if det_val <= 0:
        return 1.0

    det_term = np.sqrt(det_val)
    exp_arg = float(-(m_v - z_v).T @ iSpW @ (m_v - z_v) / 2)
    if exp_arg > 50:
        return 1.0

    exp_term = np.exp(exp_arg)
    return 1.0 - float(exp_term / det_term)


def _nearest_psd(A):
    A_sym = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def compute_state_cross_covariances(s_list, dmdm_list, H):
    cross = {}
    D0 = s_list[0].shape[0]
    for t in range(H):
        cross[(t, t)] = s_list[t].copy()
    for tau in range(1, H):
        for t in range(H - tau):
            C = s_list[t] @ dmdm_list[t]
            for k in range(t + 1, t + tau):
                C = C @ dmdm_list[k]
            cross[(t, t + tau)] = C
            cross[(t + tau, t)] = C.T
    return cross


def cost_cross_covariance(m1_aug, m2_aug, S1_aug, S2_aug, S12_aug,
                           target, Q, width):
    W = Q / (width ** 2)
    D1 = len(m1_aug)

    m1c = expected_sat_cost(m1_aug, S1_aug, target, W)
    m2c = expected_sat_cost(m2_aug, S2_aug, target, W)

    joint_S = np.block([[S1_aug, S12_aug], [S12_aug.T, S2_aug]])
    eigs = np.linalg.eigvalsh(joint_S)
    if np.min(eigs) < -1e-6:
        joint_S = _nearest_psd(joint_S)

    joint_mu = np.concatenate([m1_aug.ravel(), m2_aug.ravel()])
    joint_target = np.concatenate([target.ravel(), target.ravel()])
    joint_W = np.block([[W, np.zeros((D1, D1))], [np.zeros((D1, D1)), W]])

    m_joint_c = expected_sat_cost(joint_mu, joint_S, joint_target, joint_W)

    return (1.0 - m_joint_c) - (1.0 - m1c) * (1.0 - m2c)


def analytical_cumulative_variance(m_list, s_list, dmdm_list, cost, plant):
    angi = plant['angi']
    target_raw = np.asarray(cost['target']).ravel()
    ell = cost['p']
    D0 = len(target_raw)
    D1 = D0 + 2 * len(angi)

    target_gTrig, _, _ = gTrig(target_raw,
                                0 * np.atleast_2d(np.eye(D0)), angi,
                                compute_derivatives=False)
    target = np.concatenate([target_raw, target_gTrig.ravel()])

    Q = np.zeros((D1, D1))
    Q[np.ix_([0, D0], [0, D0])] = np.outer([1, ell], [1, ell])
    Q[D0 + 1, D0 + 1] = ell ** 2

    cw = np.atleast_1d(np.asarray(cost.get('width', [0.25]), dtype=float)).ravel()
    ncw = len(cw)

    H = len(m_list)
    m_aug = [None] * H
    s_aug = [None] * H
    for t in range(H):
        m_aug[t], s_aug[t] = augment_state(m_list[t], s_list[t], angi)

    cross = compute_state_cross_covariances(s_list, dmdm_list, H)

    var_total = 0.0
    marginal_mean = np.zeros(H)

    for t in range(H):
        mc = 0.0
        for i_idx in range(ncw):
            W = Q / (cw[i_idx] ** 2)
            mc += expected_sat_cost(m_aug[t], s_aug[t], target, W)
        marginal_mean[t] = mc / ncw

    for t in range(H):
        for t2 in range(t, H):
            w = 1.0 if t == t2 else 2.0

            if t == t2:
                cc = 0.0
                for i_idx in range(ncw):
                    W = Q / (cw[i_idx] ** 2)
                    m1c = expected_sat_cost(m_aug[t], s_aug[t], target, W)
                    cc += m1c * (1.0 - m1c)
                var_total += w * cc / ncw
            else:
                S12_raw = cross[(t, t2)]
                _, _, _, _, S12_aug = augment_cross_covariance(
                    m_list[t], m_list[t2],
                    s_list[t], s_list[t2], S12_raw, angi)

                cc = 0.0
                for i_idx in range(ncw):
                    cc += cost_cross_covariance(
                        m_aug[t], m_aug[t2],
                        s_aug[t], s_aug[t2], S12_aug,
                        target, Q, cw[i_idx])
                var_total += w * cc / ncw

    return max(var_total, 1e-12), marginal_mean


def compute_trajectory_and_stats(p, m0, S0, dynmodel, policy, plant, cost, H):
    policy['p'] = p
    D0 = len(m0)

    m = m0.copy()
    s = S0.copy()

    m_list = [m.copy()]
    s_list = [s.copy()]
    dmdm_list = []

    for t in range(H):
        result = plant['prop'](m, s, plant, dynmodel, policy,
                              compute_derivatives=True)
        m_next = result[0].copy()
        s_next = result[1].copy()
        dmdm = result[2].copy()

        m_list.append(m_next)
        s_list.append(s_next)
        dmdm_list.append(dmdm)

        m = m_next
        s = s_next

    var_c, marginal_mean = analytical_cumulative_variance(
        m_list[1:], s_list[1:], dmdm_list[1:], cost, plant)

    return m_list, s_list, dmdm_list, var_c, marginal_mean
