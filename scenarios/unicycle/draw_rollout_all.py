# draw_rollout_all.py
# *Summary:* Draw all rollouts
#
# Carl Edward Rasmussen, 2012-03-27

import numpy as np
import matplotlib.pyplot as plt
import os


def draw_rollout_all(latent, plant, t2, cost, f=None):
    """
    Draw all rollouts for the unicycle.

    Parameters:
    -----------
    latent : list of ndarray
        List of latent state matrices from rollouts
    plant : dict
        Plant structure
    t2 : float
        Supersampling time step
    cost : dict
        Cost structure
    f : str or None
        Optional filename prefix for saving frames
    """
    try:
        from .draw_unicycle import draw_unicycle
    except ImportError:
        from draw_unicycle import draw_unicycle

    plt.clf()

    J = 5
    t1 = plant['dt']

    if f is not None and f == '':
        f = 'tmp/tmp'

    rw = 0.225    # wheel radius
    rf = 0.54     # frame center of mass to wheel
    rt = 0.27     # frame centre of mass to turntable
    rr = rf + rt  # distance wheel to turntable

    M = 24
    MM = 2 * np.pi * np.arange(M + 1) / M
    RR_list = ['r-', 'r-', 'r-', 'k-', 'b-', 'b-', 'b-']
    ii_counter = 10000

    for jj in range(len(latent)):
        qq = latent[jj]

        # Interpolate
        xi = t1 * np.arange(qq.shape[0])
        xn = np.arange(0, (qq.shape[0] - 1) * t1 + t2, t2)
        q = np.zeros((len(xn), qq.shape[1]))
        for i_col in range(qq.shape[1]):
            q[:, i_col] = np.interp(xn, xi, qq[:, i_col])

        for i_frame in range(q.shape[0]):
            xpos = q[i_frame, 9]
            ypos = q[i_frame, 10]
            theta = q[i_frame, 13]
            phi = q[i_frame, 14]
            psiw = -q[i_frame, 15]
            psif = q[i_frame, 16]
            psit = q[i_frame, 17]

            A_mat = np.array([
                [np.cos(phi), np.sin(phi), 0.0],
                [-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta), -np.sin(theta)],
                [-np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta), np.cos(theta)]
            ]).T

            # R{1}: wheel circle
            r_in = rw * np.vstack([np.cos(psiw + MM), np.zeros(M + 1), np.sin(psiw + MM) + 1.0])
            R1 = A_mat @ r_in + np.tile(np.array([xpos, ypos, 0.0]).reshape(-1, 1), (1, M + 1))

            # R{2}: wheel diameter
            r_in = rw * np.array([[np.cos(psiw), -np.cos(psiw)],
                                   [0.0, 0.0],
                                   [np.sin(psiw) + 1.0, -np.sin(psiw) + 1.0]])
            R2 = A_mat @ r_in + np.tile(np.array([[xpos], [ypos], [0.0]]), (1, 2))

            # R{3}: wheel perpendicular
            r_in = rw * np.array([[np.sin(psiw), -np.sin(psiw)],
                                   [0.0, 0.0],
                                   [-np.cos(psiw) + 1.0, np.cos(psiw) + 1.0]])
            R3 = A_mat @ r_in + np.tile(np.array([[xpos], [ypos], [0.0]]), (1, 2))

            # R{4}: fork
            r_in = np.array([[0.0, rr * np.sin(psif)],
                              [0.0, 0.0],
                              [rw, rw + rr * np.cos(psif)]])
            R4 = A_mat @ r_in + np.tile(np.array([[xpos], [ypos], [0.0]]), (1, 2))

            # R{5}: turntable circle
            r_in = np.vstack([rr * np.sin(psif) + rw * np.cos(psif) * np.cos(psit + MM),
                               rw * np.sin(psit + MM),
                               rw + rr * np.cos(psif) - rw * np.sin(psif) * np.cos(psit + MM)])
            R5 = A_mat @ r_in + np.tile(np.array([xpos, ypos, 0.0]).reshape(-1, 1), (1, M + 1))

            # R{6}: turntable diameter
            r_in = np.array([[rr * np.sin(psif) + rw * np.cos(psif) * np.cos(psit),
                               rr * np.sin(psif) - rw * np.cos(psif) * np.cos(psit)],
                              [rw * np.sin(psit), -rw * np.sin(psit)],
                              [rw + rr * np.cos(psif) - rw * np.sin(psif) * np.cos(psit),
                               rw + rr * np.cos(psif) + rw * np.sin(psif) * np.cos(psit)]])
            R6 = A_mat @ r_in + np.tile(np.array([[xpos], [ypos], [0.0]]), (1, 2))

            # R{7}: turntable perpendicular
            r_in = np.array([[rr * np.sin(psif) + rw * np.cos(psif) * np.sin(psit),
                               rr * np.sin(psif) - rw * np.cos(psif) * np.sin(psit)],
                              [-rw * np.cos(psit), rw * np.cos(psit)],
                              [rw + rr * np.cos(psif) - rw * np.sin(psif) * np.sin(psit),
                               rw + rr * np.cos(psif) + rw * np.sin(psif) * np.sin(psit)]])
            R7 = A_mat @ r_in + np.tile(np.array([[xpos], [ypos], [0.0]]), (1, 2))

            plt.clf()
            plt.gca()
            ax = plt.gcf().add_subplot(111, projection='3d')

            # Reference circle
            aa = np.linspace(0, 2 * np.pi, 201)
            ax.plot(2 * np.sin(aa), 2 * np.cos(aa), 0 * aa, 'k:', linewidth=2)

            # Filled patches for wheel
            r_center = A_mat @ np.array([0.0, 0.0, rw]) + np.array([xpos, ypos, 0.0])
            mq1 = int(M / 4) + 1
            P_x = np.concatenate([[r_center[0]], R1[0, :mq1], [r_center[0]]])
            P_y = np.concatenate([[r_center[1]], R1[1, :mq1], [r_center[1]]])
            P_z = np.concatenate([[r_center[2]], R1[2, :mq1], [r_center[2]]])
            ax.plot_trisurf(P_x, P_y, P_z, color='r', edgecolor='none', alpha=0.5)

            mq2 = int(M / 2) + 1
            mq3 = int(3 * M / 4) + 1
            P_x = np.concatenate([[r_center[0]], R1[0, mq2:mq3], [r_center[0]]])
            P_y = np.concatenate([[r_center[1]], R1[1, mq2:mq3], [r_center[1]]])
            P_z = np.concatenate([[r_center[2]], R1[2, mq2:mq3], [r_center[2]]])
            ax.plot_trisurf(P_x, P_y, P_z, color='r', edgecolor='none', alpha=0.5)

            # Filled patches for turntable
            r_turntable = A_mat @ np.array([rr * np.sin(psif), 0.0, rw + rr * np.cos(psif)]) + np.array([xpos, ypos, 0.0])
            P_x = np.concatenate([[r_turntable[0]], R5[0, :mq1], [r_turntable[0]]])
            P_y = np.concatenate([[r_turntable[1]], R5[1, :mq1], [r_turntable[1]]])
            P_z = np.concatenate([[r_turntable[2]], R5[2, :mq1], [r_turntable[2]]])
            ax.plot_trisurf(P_x, P_y, P_z, color='b', edgecolor='none', alpha=0.5)

            P_x = np.concatenate([[r_turntable[0]], R5[0, mq2:mq3], [r_turntable[0]]])
            P_y = np.concatenate([[r_turntable[1]], R5[1, mq2:mq3], [r_turntable[1]]])
            P_z = np.concatenate([[r_turntable[2]], R5[2, mq2:mq3], [r_turntable[2]]])
            ax.plot_trisurf(P_x, P_y, P_z, color='b', edgecolor='none', alpha=0.5)

            # Lines
            R_all = [R1, R2, R3, R4, R5, R6, R7]
            for j_idx in [0, 3, 4]:
                ax.plot(R_all[j_idx][0, :], R_all[j_idx][1, :],
                        R_all[j_idx][2, :], RR_list[j_idx], linewidth=2)

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(0, 1.5)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.grid(True)

            # draw controls
            ut = q[i_frame, -2]
            uw = q[i_frame, -1]

            dyno_indices = plant['dyno']
            s_state = q[i_frame, dyno_indices].ravel()
            L_cost = cost['fcn'](cost, s_state, np.zeros(len(dyno_indices)))
            if isinstance(L_cost, tuple):
                L_val = L_cost[0]
            else:
                L_val = L_cost

            utM = 10.0
            uwM = 50.0

            oo = np.array([4.0, -3.07, 0.0]) / 6.4
            o1 = np.array([-0.5, 2.0, 2.0])
            o2 = np.array([-0.5, 2.0, 1.6])
            o3 = np.array([-0.5, 2.0, 1.2])

            o0 = 1.5 * ut / utM
            ax.plot([o1[0], o1[0] + o0 * oo[0]],
                    [o1[1], o1[1] + o0 * oo[1]],
                    [o1[2], o1[2] + o0 * oo[2]], 'b', linewidth=5)
            ax.plot([o1[0] - 1.5 * oo[0], o1[0] + 1.5 * oo[0],
                      o1[0] + 1.5 * oo[0], o1[0] - 1.5 * oo[0],
                      o1[0] - 1.5 * oo[0]],
                     [o1[1] - 1.5 * oo[1], o1[1] + 1.5 * oo[1],
                      o1[1] + 1.5 * oo[1], o1[1] - 1.5 * oo[1],
                      o1[1] - 1.5 * oo[1]],
                     [o1[2] + 0.04, o1[2] + 0.04,
                      o1[2] - 0.04, o1[2] - 0.04,
                      o1[2] + 0.04], 'b')
            ax.plot([-0.5, -0.5], [2, 2], [o1[2] - 0.06, o1[2] + 0.06], 'b')

            o0 = 1.5 * uw / uwM
            ax.plot([o2[0], o2[0] + o0 * oo[0]],
                    [o2[1], o2[1] + o0 * oo[1]],
                    [o2[2], o2[2] + o0 * oo[2]], 'r', linewidth=5)
            ax.plot([o2[0] - 1.5 * oo[0], o2[0] + 1.5 * oo[0],
                      o2[0] + 1.5 * oo[0], o2[0] - 1.5 * oo[0],
                      o2[0] - 1.5 * oo[0]],
                     [o2[1] - 1.5 * oo[1], o2[1] + 1.5 * oo[1],
                      o2[1] + 1.5 * oo[1], o2[1] - 1.5 * oo[1],
                      o2[1] - 1.5 * oo[1]],
                     [o2[2] + 0.04, o2[2] + 0.04,
                      o2[2] - 0.04, o2[2] - 0.04,
                      o2[2] + 0.04], 'r')
            ax.plot([-0.5, -0.5], [2, 2], [o2[2] - 0.06, o2[2] + 0.06], 'r')

            o0 = 3.0 * L_val - 1.5
            ax.plot([o3[0] - 1.5 * oo[0], o3[0] + o0 * oo[0]],
                    [o3[1] - 1.5 * oo[1], o3[1] + o0 * oo[1]],
                    [o3[2] - 1.5 * oo[2], o3[2] + o0 * oo[2]], 'k', linewidth=5)
            ax.plot([o3[0] - 1.5 * oo[0], o3[0] + 1.5 * oo[0],
                      o3[0] + 1.5 * oo[0], o3[0] - 1.5 * oo[0],
                      o3[0] - 1.5 * oo[0]],
                     [o3[1] - 1.5 * oo[1], o3[1] + 1.5 * oo[1],
                      o3[1] + 1.5 * oo[1], o3[1] - 1.5 * oo[1],
                      o3[1] - 1.5 * oo[1]],
                     [o3[2] + 0.04, o3[2] + 0.04,
                      o3[2] - 0.04, o3[2] - 0.04,
                      o3[2] + 0.04], 'k')

            ax.text(-0.5 - 1.5 * oo[0], 2 - 1.5 * oo[1], 2.2,
                    'Disc torque    max pm 10 Nm', color='b', fontsize=16)
            ax.text(-0.5 - 1.5 * oo[0], 2 - 1.5 * oo[1], 1.8,
                    'Wheel torque max pm 50 Nm', color='r', fontsize=16)
            ax.text(-0.5 - 1.5 * oo[0], 2 - 1.5 * oo[1], 1.4,
                    'Instantaneous Cost', color='k', fontsize=16)

            if jj >= J:
                exx = 0
                for exxx in range(jj):
                    exx += latent[exxx].shape[0] - 1
                ax.text(2, 1, 1.8, 'Control trial #%d' % (jj - J + 1), fontsize=16)
                ax.text(2, 1, 1.4, 'Experience: %2.1f s' % (exx * t1), fontsize=16)
            else:
                ax.text(2, 1, 1.8, 'Random trial #%d' % (jj + 1), fontsize=16)

            plt.pause(0.05)

    return ax
