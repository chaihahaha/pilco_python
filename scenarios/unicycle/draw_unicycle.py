# draw_unicycle.py
# *Summary:* Draw the unicycle with cost and applied torques
#
#    def draw_unicycle(latent, plant, t2, cost, text1=None, text2=None)
#
# *Input arguments:*
#
#   latent     state of the unicycle (including the torques)
#   plant      plant structure (dict)
#     .dt      sampling time
#     .dyno    state indices that are passed on to the cost function
#   t2         supersampling frequency (for smoother plots)
#   cost       cost structure (dict)
#     .fcn     function handle (it is assumed to use saturating cost)
#   text1      (optional) text field 1
#   text2      (optional) text field 2
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
#
# Last modified: 2013-04-04

import numpy as np
import matplotlib.pyplot as plt


def draw_unicycle(latent, plant, t2, cost, text1=None, text2=None):
    ## Code

    plt.clf()
    ax = plt.gcf().add_subplot(111, projection='3d')

    t1 = plant['dt']

    rw = 0.225    # wheel radius
    rf = 0.54     # frame center of mass to wheel
    rt = 0.27     # frame centre of mass to turntable
    rr = rf + rt  # distance wheel to turntable

    M = 24
    MM = 2 * np.pi * np.arange(M + 1) / M
    RR_list = ['r-', 'r-', 'r-', 'k-', 'b-', 'b-', 'b-']

    qq = np.atleast_2d(np.asarray(latent))

    # Interpolate for smoother plots
    xi = t1 * np.arange(qq.shape[0])
    xn = np.arange(0, (qq.shape[0] - 1) * t1 + t2, t2)
    q = np.zeros((len(xn), qq.shape[1]))
    for i_col in range(qq.shape[1]):
        q[:, i_col] = np.interp(xn, xi, qq[:, i_col])

    for i_frame in range(q.shape[0]):
        # State indices (0-based) in the full 20-element state:
        # 0:dx, 1:dy, 2:dxc, 3:dyc, 4:dtheta, 5:dphi, 6:dpsiw, 7:dpsif, 8:dpsit,
        # 9:x, 10:y, 11:xc, 12:yc, 13:theta, 14:phi, 15:psiw, 16:psif, 17:psit,
        # 18:ct, 19:cw
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
        R1 = A_mat @ r_in + np.column_stack([np.full(len(MM), xpos),
                                               np.full(len(MM), ypos),
                                               np.zeros(len(MM))]).T

        # R{2}: wheel diameter lines
        r_in = rw * np.array([[np.cos(psiw), -np.cos(psiw)],
                               [0.0, 0.0],
                               [np.sin(psiw) + 1.0, -np.sin(psiw) + 1.0]])
        R2 = A_mat @ r_in + np.array([[xpos, xpos], [ypos, ypos], [0.0, 0.0]])

        # R{3}: wheel perpendicular diameter
        r_in = rw * np.array([[np.sin(psiw), -np.sin(psiw)],
                               [0.0, 0.0],
                               [-np.cos(psiw) + 1.0, np.cos(psiw) + 1.0]])
        R3 = A_mat @ r_in + np.array([[xpos, xpos], [ypos, ypos], [0.0, 0.0]])

        # R{4}: fork line
        r_in = np.array([[0.0, rr * np.sin(psif)],
                          [0.0, 0.0],
                          [rw, rw + rr * np.cos(psif)]])
        R4 = A_mat @ r_in + np.array([[xpos, xpos], [ypos, ypos], [0.0, 0.0]])

        # R{5}: turntable circle
        r_in = np.vstack([rr * np.sin(psif) + rw * np.cos(psif) * np.cos(psit + MM),
                           rw * np.sin(psit + MM),
                           rw + rr * np.cos(psif) - rw * np.sin(psif) * np.cos(psit + MM)])
        R5 = A_mat @ r_in + np.column_stack([np.full(len(MM), xpos),
                                               np.full(len(MM), ypos),
                                               np.zeros(len(MM))]).T

        # R{6}: turntable diameter
        r_in = np.array([[rr * np.sin(psif) + rw * np.cos(psif) * np.cos(psit),
                           rr * np.sin(psif) - rw * np.cos(psif) * np.cos(psit)],
                          [rw * np.sin(psit), -rw * np.sin(psit)],
                          [rw + rr * np.cos(psif) - rw * np.sin(psif) * np.cos(psit),
                           rw + rr * np.cos(psif) + rw * np.sin(psif) * np.cos(psit)]])
        R6 = A_mat @ r_in + np.array([[xpos, xpos], [ypos, ypos], [0.0, 0.0]])

        # R{7}: turntable perpendicular diameter
        r_in = np.array([[rr * np.sin(psif) + rw * np.cos(psif) * np.sin(psit),
                           rr * np.sin(psif) - rw * np.cos(psif) * np.sin(psit)],
                          [-rw * np.cos(psit), rw * np.cos(psit)],
                          [rw + rr * np.cos(psif) - rw * np.sin(psif) * np.sin(psit),
                           rw + rr * np.cos(psif) + rw * np.sin(psif) * np.sin(psit)]])
        R7 = A_mat @ r_in + np.array([[xpos, xpos], [ypos, ypos], [0.0, 0.0]])

        ax.cla()

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

        # Plot lines
        R_all = [R1, R2, R3, R4, R5, R6, R7]
        for j_idx in [0, 3, 4]:  # indices [1,4,5] in MATLAB -> [0,3,4] in Python
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
                'Disc torque    max pm 10 Nm', color='b', fontsize=10)
        ax.text(-0.5 - 1.5 * oo[0], 2 - 1.5 * oo[1], 1.8,
                'Wheel torque max pm 50 Nm', color='r', fontsize=10)
        ax.text(-0.5 - 1.5 * oo[0], 2 - 1.5 * oo[1], 1.4,
                'Instantaneous Cost', color='k', fontsize=10)

        if text1 is not None:
            ax.text(2, 1, 1.8, text1, fontsize=10)
        if text2 is not None:
            ax.text(2, 1, 1.4, text2, fontsize=10)

        plt.pause(0.05)

    return ax
