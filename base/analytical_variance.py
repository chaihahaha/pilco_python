import numpy as np
from pilco_python.util.unwrap import unwrap
from pilco_python.util.rewrap import rewrap


def augment_state(mu, S, angi):
    D0 = len(mu)
    D1 = D0 + 2 * len(angi)
    mu_aug = np.zeros(D1); mu_aug[:D0] = mu.ravel()
    if len(angi) == 0:
        S_aug = np.zeros((D1, D1)); S_aug[:D0, :D0] = S
        return mu_aug, S_aug
    from pilco_python.util.gTrig import gTrig
    m_g, s_g, C = gTrig(mu.ravel(), np.atleast_2d(S), angi, compute_derivatives=False)
    mu_aug[D0:] = m_g.ravel(); S_aug = np.zeros((D1, D1)); S_aug[:D0, :D0] = S
    S_aug[D0:, D0:] = s_g; cross = S @ C
    S_aug[:D0, D0:] = cross; S_aug[D0:, :D0] = cross.T
    return mu_aug, S_aug


def augment_cross_covariance(mu1, mu2, S11, S22, S12, angi):
    D0 = len(mu1); D1 = D0 + 2 * len(angi)
    mu_a1, mu_a2 = np.zeros(D1), np.zeros(D1)
    mu_a1[:D0], mu_a2[:D0] = mu1.ravel(), mu2.ravel()
    S_a11, S_a22, S_a12 = np.zeros((D1, D1)), np.zeros((D1, D1)), np.zeros((D1, D1))
    S_a11[:D0, :D0], S_a22[:D0, :D0], S_a12[:D0, :D0] = S11, S22, S12

    for ai, ang_idx in enumerate(angi):
        aa = D0 + 2 * ai
        m_i, m_j = mu1[ang_idx], mu2[ang_idx]
        v_i = S11[ang_idx, ang_idx]; v_j = S22[ang_idx, ang_idx]; v_ij = S12[ang_idx, ang_idx]
        ci, cj = np.exp(-v_i / 2), np.exp(-v_j / 2)
        mu_a1[aa] = np.sin(m_i) * ci; mu_a1[aa + 1] = np.cos(m_i) * ci
        mu_a2[aa] = np.sin(m_j) * cj; mu_a2[aa + 1] = np.cos(m_j) * cj

        dm, sm = m_i - m_j, m_i + m_j
        vd, vs = v_i + v_j - 2 * v_ij, v_i + v_j + 2 * v_ij
        ed, es = np.exp(-vd / 2), np.exp(-vs / 2)
        E_ss = 0.5 * (np.cos(dm) * ed - np.cos(sm) * es)
        E_cc = 0.5 * (np.cos(dm) * ed + np.cos(sm) * es)
        E_sc = 0.5 * (np.sin(sm) * es + np.sin(dm) * ed)
        E_cs = 0.5 * (np.sin(sm) * es - np.sin(dm) * ed)

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

        cos_i, ns_i = np.cos(m_i) * ci, -np.sin(m_i) * ci
        cos_j, ns_j = np.cos(m_j) * cj, -np.sin(m_j) * cj
        for b in range(D0):
            if b == ang_idx: continue
            cv_i = S12[b, ang_idx]
            S_a12[b, aa] = cv_i * cos_j; S_a12[b, aa + 1] = cv_i * ns_j
            S_a12[aa, b] = cv_i * cos_i; S_a12[aa + 1, b] = cv_i * ns_i
        S_a11[aa, ang_idx] = v_i * cos_i - mu_a1[aa] * mu_a1[ang_idx]
        S_a11[aa + 1, ang_idx] = v_i * ns_i - mu_a1[aa + 1] * mu_a1[ang_idx]
        S_a11[ang_idx, aa] = S_a11[aa, ang_idx]; S_a11[ang_idx, aa + 1] = S_a11[aa + 1, ang_idx]
        S_a22[aa, ang_idx] = v_j * cos_j - mu_a2[aa] * mu_a2[ang_idx]
        S_a22[aa + 1, ang_idx] = v_j * ns_j - mu_a2[aa + 1] * mu_a2[ang_idx]
        S_a22[ang_idx, aa] = S_a22[aa, ang_idx]; S_a22[ang_idx, aa + 1] = S_a22[aa + 1, ang_idx]

    for ai_i, ang_i in enumerate(angi):
        a_i = D0 + 2 * ai_i
        for aj_j, ang_j in enumerate(angi):
            a_j = D0 + 2 * aj_j
            m_ii, m_jj = mu1[ang_i], mu2[ang_j]
            v_ii, v_jj = S11[ang_i, ang_i], S22[ang_j, ang_j]
            v_ij = S12[ang_i, ang_j]
            ci, cj = np.exp(-v_ii / 2), np.exp(-v_jj / 2)
            dm, sm = m_ii - m_jj, m_ii + m_jj
            vd, vs = v_ii + v_jj - 2 * v_ij, v_ii + v_jj + 2 * v_ij
            ed, es = np.exp(-vd / 2), np.exp(-vs / 2)
            E_si, E_ci = np.sin(m_ii) * ci, np.cos(m_ii) * ci
            E_sj, E_cj = np.sin(m_jj) * cj, np.cos(m_jj) * cj
            S_a12[a_i, a_j] = 0.5 * (np.cos(dm) * ed - np.cos(sm) * es) - E_si * E_sj
            S_a12[a_i + 1, a_j] = 0.5 * (np.sin(sm) * es - np.sin(dm) * ed) - E_ci * E_sj
            S_a12[a_i, a_j + 1] = 0.5 * (np.sin(sm) * es + np.sin(dm) * ed) - E_si * E_cj
            S_a12[a_i + 1, a_j + 1] = 0.5 * (np.cos(dm) * ed + np.cos(sm) * es) - E_ci * E_cj
            S_a12[a_j, a_i] = S_a12[a_i, a_j]
            S_a12[a_j + 1, a_i] = S_a12[a_i + 1, a_j]
            S_a12[a_j, a_i + 1] = S_a12[a_i, a_j + 1]
            S_a12[a_j + 1, a_i + 1] = S_a12[a_i + 1, a_j + 1]

    return mu_a1, mu_a2, S_a11, S_a22, S_a12


def expected_sat_cost(mu, Sigma, z, W):
    D = len(mu); In = np.eye(D); m_v = mu.reshape(-1, 1); z_v = z.reshape(-1, 1)
    M = In + Sigma @ W
    try:
        iSpW = np.linalg.solve(M.T, W.T).T; det_val = np.linalg.det(M)
    except np.linalg.LinAlgError:
        iSpW = np.linalg.lstsq(M.T, W.T, rcond=None)[0].T
        det_val = np.linalg.det(M + np.eye(D) * 1e-8)
    if det_val <= 1e-15: return 1.0
    exp_arg = float(-(m_v - z_v).T @ iSpW @ (m_v - z_v) / 2)
    if exp_arg > 50 or np.isnan(exp_arg): return 1.0
    result = 1.0 - float(np.exp(exp_arg) / np.sqrt(max(det_val, 1e-15)))
    if np.isnan(result): return 1.0
    return result


def _nearest_psd(A):
    A_sym = (A + A.T) / 2; eigvals, eigvecs = np.linalg.eigh(A_sym)
    return eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.T


def cost_cross_covariance(m1_aug, m2_aug, S1_aug, S2_aug, S12_aug, target, Q, width):
    W = Q / (width ** 2); D1 = len(m1_aug)
    m1c = expected_sat_cost(m1_aug, S1_aug, target, W)
    m2c = expected_sat_cost(m2_aug, S2_aug, target, W)
    joint_S = np.block([[S1_aug, S12_aug], [S12_aug.T, S2_aug]])
    if np.min(np.linalg.eigvalsh(joint_S)) < -1e-6:
        joint_S = _nearest_psd(joint_S)
    joint_mu = np.concatenate([m1_aug.ravel(), m2_aug.ravel()])
    joint_target = np.concatenate([target.ravel(), target.ravel()])
    joint_W = np.block([[W, np.zeros((D1, D1))], [np.zeros((D1, D1)), W]])
    m_joint_c = expected_sat_cost(joint_mu, joint_S, joint_target, joint_W)
    return (1.0 - m_joint_c) - (1.0 - m1c) * (1.0 - m2c)


def compute_state_cross_covariances(s_list, dmdm_list, H):
    cross = {}
    for t in range(H): cross[(t, t)] = s_list[t].copy()
    for tau in range(1, H):
        for t in range(H - tau):
            C = s_list[t] @ dmdm_list[t]
            for k in range(t + 1, t + tau): C = C @ dmdm_list[k]
            cross[(t, t + tau)] = C; cross[(t + tau, t)] = C.T
    return cross


def _build_cost_Q(cost, D0, D1, ell, target):
    ell_arr = np.asarray(ell).ravel()
    if ell_arr.size == 1:
        ell_val = float(ell_arr[0])
        Q = np.zeros((D1, D1))
        Q[np.ix_([0, D0], [0, D0])] = np.outer([1, ell_val], [1, ell_val])
        Q[D0 + 1, D0 + 1] = ell_val ** 2
    else:
        el1, el2 = float(ell_arr[0]), float(ell_arr[1])
        C_mat = np.array([[1, -el1, 0, -el2, 0], [0, 0, el1, 0, el2]])
        Q = np.zeros((D1, D1)); idx_q = [0, D0, D0 + 1, D0 + 2, D0 + 3]
        CtC = C_mat.T @ C_mat
        for pi, ri in enumerate(idx_q):
            for pj, rj in enumerate(idx_q): Q[ri, rj] = CtC[pi, pj]
    return Q


def analytical_cumulative_variance(m_list, s_list, dmdm_list, cost, plant):
    angi = plant['angi']
    target_raw = np.asarray(cost['target']).ravel()
    ell = cost['p']; D0 = len(target_raw); D1 = D0 + 2 * len(angi)
    from pilco_python.util.gTrig import gTrig
    tg, _, _ = gTrig(target_raw, 0 * np.atleast_2d(np.eye(D0)), angi, compute_derivatives=False)
    target = np.concatenate([target_raw, tg.ravel()])
    Q = _build_cost_Q(cost, D0, D1, ell, target)
    cw = np.atleast_1d(np.asarray(cost.get('width', [0.25] if (np.asarray(ell).ravel().size == 1) else [0.5]), dtype=float)).ravel()
    ncw = len(cw)
    H = len(m_list)
    m_aug = [augment_state(m_list[t], s_list[t], angi)[0] for t in range(H)]
    s_aug = [augment_state(m_list[t], s_list[t], angi)[1] for t in range(H)]
    cross = compute_state_cross_covariances(s_list, dmdm_list, H)
    var_total = 0.0
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
                    m_list[t], m_list[t2], s_list[t], s_list[t2], S12_raw, angi)
                cc = 0.0
                for i_idx in range(ncw):
                    cc += cost_cross_covariance(m_aug[t], m_aug[t2], s_aug[t], s_aug[t2], S12_aug, target, Q, cw[i_idx])
                var_total += w * cc / ncw
    return max(var_total, 1e-12)


def compute_trajectory_and_stats(p, m0, S0, dynmodel, policy, plant, cost, H):
    policy['p'] = p; D0 = len(m0)
    m, s = m0.copy(), S0.copy()
    m_list, s_list = [m.copy()], [s.copy()]
    dmdm_list = []
    for t in range(H):
        result = plant['prop'](m, s, plant, dynmodel, policy, compute_derivatives=True)
        m_next = result[0].copy(); s_next = result[1].copy(); dmdm = result[2].copy()
        m_list.append(m_next); s_list.append(s_next); dmdm_list.append(dmdm)
        m, s = m_next, s_next
    var_c = analytical_cumulative_variance(m_list[1:], s_list[1:], dmdm_list[1:], cost, plant)
    return m_list, s_list, dmdm_list, var_c
