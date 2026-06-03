# dynamics_unicycle.py
# *Summary:* Implements the ODE for simulating the unicycle dynamics.
#
#    def dynamics_unicycle(t, z, u)
#
# *Input arguments:*
#
#   t     current time step (called from ODE solver)
#   z     state                                                    [12]
#   u     control torques [V, U] where V=turntable, U=wheel        [2]
#
# *Output arguments:*
#
#   dz    state derivative wrt time
#
# Note: It is assumed that the state variables are of the following order:
# state: z = [dtheta, dphi, dpsiw, dpsif, dpsit,
#             x,  y,  theta,  phi,  psiw,  psif,  psit]
#
#   theta: tilt of the unicycle
#   phi: orientation of the unicycle
#   psiw: angle of wheel (rotation)
#   psif: angle of fork
#   psit: angle of turntable (rotation)
#
#       dtheta   angular velocity of tilt of the unicycle
#       dphi     angular velocity of orientation of the unicycle
#       dpsiw    angular velocity of wheel
#       dpsif    angular velocity of fork
#       dpsit    angular velocity of turntable
#       x        x-position of contact point in plane
#       y        y-position of contact point in plane
#       theta    tilt of the unicycle
#       phi      orientation of the unicycle
#       psiw     angle of wheel (rotation)
#       psif     angle of fork
#       psit     angle of turntable (rotation)
#
#
# Copyright (C) 2008-2013 by
# Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen,
# based on derivations by David Forster
#
# Last modified: 2013-03-18

import numpy as np


def dynamics_unicycle(t, z, u):
    ## Code

    z = np.asarray(z, dtype=np.float64).ravel()
    u = np.asarray(u).ravel()

    T_fric = 0.0  # no friction

    # State variables (0-indexed)
    dtheta = z[0]; dphi = z[1]; dpsiw = z[2]; dpsif = z[3]; dpsit = z[4]
    x = z[5]; y = z[6]; theta = z[7]; phi = z[8]
    # psiw = z[9] and psit = z[11] are not needed (cleared in MATLAB)
    psif = z[10]

    # plant characteristics
    mt = 10.0    # turntable mass
    mw = 1.0     # wheel mass
    mf = 23.5    # frame mass
    rw = 0.225   # wheel radius
    rf = 0.54    # frame center of mass to wheel
    rt = 0.27    # frame centre of mass to turntable
    r = rf + rt  # distance wheel to turntable
    Cw = 0.0484  # moment of inertia of wheel around axle
    Aw = 0.0242  # moment of inertia of wheel perpendicular to axle
    Cf = 0.8292  # moment of inertia of frame
    Bf = 0.4608  # moment of inertia of frame
    Af = 0.4248  # moment of inertia of frame
    Ct = 0.2     # moment of inertia of turntable around axle
    At = 1.3     # moment of inertia of turntable perpendicular to axle
    g = 9.82     # acceleration of gravity

    st = np.sin(theta); ct = np.cos(theta)
    sf = np.sin(psif); cf = np.cos(psif)

    # control inputs
    V_val = u[0]  # turntable torque
    U_val = u[1]  # wheel torque

    # ---- Matrix A (5x5) ----
    A = np.zeros((5, 5))

    # Row 0 (MATLAB row 1)
    A[0, 0] = -Ct * sf
    A[0, 1] = Ct * cf * ct
    A[0, 2] = 0.0
    A[0, 3] = 0.0
    A[0, 4] = Ct

    # Row 1 (MATLAB row 2)
    A[1, 0] = 0.0
    A[1, 1] = (Cw * st + At * st
               - rf * (-mf * (st * rf + cf * st * rw)
                       - mt * (st * r + cf * st * rw))
               + rt * mt * (st * r + cf * st * rw))
    A[1, 2] = -cf * rw * (rf * (mf + mt) + rt * mt)
    A[1, 3] = -Cw - At - rf * (mf * rf + mt * r) - rt * mt * r
    A[1, 4] = 0.0

    # Row 2 (MATLAB row 3)
    A[2, 0] = (cf * (-Af * sf - Ct * sf)
               - sf * (-Bf * cf - At * cf
                       + rf * (-mf * (cf * rf + rw) - mt * (cf * r + rw))
                       - rt * mt * (cf * r + rw)))
    A[2, 1] = (Aw * ct
               + cf * (Af * cf * ct + Ct * cf * ct)
               - sf * (-Bf * sf * ct - At * sf * ct
                       + rf * (-mf * sf * ct * rf - mt * sf * ct * r)
                       - rt * mt * sf * ct * r))
    A[2, 2] = 0.0
    A[2, 3] = 0.0
    A[2, 4] = Ct * cf

    # Row 3 (MATLAB row 4)
    A[3, 0] = (-Aw
               - rw * (mf * (cf * rf + rw) + mw * rw + mt * (cf * r + rw))
               + sf * (-Af * sf - Ct * sf)
               + cf * (-Bf * cf - At * cf
                       + rf * (-mf * (cf * rf + rw) - mt * (cf * r + rw))
                       - rt * mt * (cf * r + rw)))
    A[3, 1] = (-rw * (mt * sf * ct * r + mf * sf * ct * rf)
               + sf * (Af * cf * ct + Ct * cf * ct)
               + cf * (-Bf * sf * ct - At * sf * ct
                       + rf * (-mf * sf * ct * rf - mt * sf * ct * r)
                       - rt * mt * sf * ct * r))
    A[3, 2] = 0.0
    A[3, 3] = 0.0
    A[3, 4] = Ct * sf

    # Row 4 (MATLAB row 5)
    A[4, 0] = 0.0
    A[4, 1] = (2 * Cw * st + At * st
               - rf * (-mt * (st * r + cf * st * rw)
                       - mf * (st * rf + cf * st * rw))
               + rt * mt * (st * r + cf * st * rw)
               + rw * (mw * st * rw
                       + sf * (mf * sf * st * rw + mt * sf * st * rw)
                       + cf * (mt * (st * r + cf * st * rw)
                               + mf * (st * rf + cf * st * rw))))
    A[4, 2] = (-Cw
               - rt * mt * cf * rw
               + rw * (-mw * rw
                       + sf * (-mf * sf * rw - mt * sf * rw)
                       + cf * (-mf * cf * rw - mt * cf * rw))
               - rf * (mt * cf * rw + mf * cf * rw))
    A[4, 3] = (-Cw - At
               - rf * (mf * rf + mt * r)
               - rt * mt * r
               - rw * cf * (mf * rf + mt * r))
    A[4, 4] = 0.0

    # ---- Vector b (5x1) ----
    b = np.zeros(5)

    # b[0] (MATLAB b(1))
    b[0] = (-V_val
            + Ct * (-dphi * sf * dpsif * ct
                    - dphi * cf * st * dtheta
                    - cf * dpsif * dtheta))

    # b[1] (MATLAB b(2))
    term_rf_inner = (-mf * g * sf * ct
                     - mf * (sf * dpsif * (-dphi * st + dpsiw) * rw
                             + cf * dphi * ct * dtheta * rw
                             - (-dphi * cf * ct + sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * rf)
                             + dphi * ct * dtheta * rf
                             - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw)
                     - mt * g * sf * ct
                     - mt * (sf * dpsif * (-dphi * st + dpsiw) * rw
                             + cf * dphi * ct * dtheta * rw
                             - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw
                             + dphi * ct * dtheta * (rf + rt)
                             + (dphi * cf * ct - sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt))))
    term_rt_inner = (-mt * g * sf * ct
                     - mt * (sf * dpsif * (-dphi * st + dpsiw) * rw
                             + cf * dphi * ct * dtheta * rw
                             - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw
                             + dphi * ct * dtheta * (rf + rt)
                             + (dphi * cf * ct - sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt))))

    b[1] = (-U_val
            + Cw * dphi * ct * dtheta
            - (-dphi * cf * ct + sf * dtheta) * Bf * (dphi * sf * ct + cf * dtheta)
            + (dphi * sf * ct + cf * dtheta) * Af * (-dphi * cf * ct + sf * dtheta)
            + At * dphi * ct * dtheta
            - (dphi * sf * ct + cf * dtheta) * Ct * (dphi * cf * ct - sf * dtheta + dpsit)
            + (dphi * cf * ct - sf * dtheta) * At * (dphi * sf * ct + cf * dtheta)
            - rf * term_rf_inner
            - rt * term_rt_inner)

    # b[2] (MATLAB b(3))
    term_b3_cf_inner = (-Af * (dphi * sf * dpsif * ct
                              + dphi * cf * st * dtheta
                              + cf * dpsif * dtheta)
                        - (dphi * sf * ct + cf * dtheta) * Cf * (-dphi * st + dpsif)
                        + (-dphi * st + dpsif) * Bf * (dphi * sf * ct + cf * dtheta)
                        + Ct * (-dphi * sf * dpsif * ct
                                - dphi * cf * st * dtheta
                                - cf * dpsif * dtheta))
    term_b3_sf_inner = (-Bf * (dphi * cf * dpsif * ct
                              - dphi * sf * st * dtheta
                              - dpsif * sf * dtheta)
                        - (-dphi * st + dpsif) * Af * (-dphi * cf * ct + sf * dtheta)
                        + (-dphi * cf * ct + sf * dtheta) * Cf * (-dphi * st + dpsif)
                        - At * (dphi * cf * dpsif * ct
                                - dphi * sf * st * dtheta
                                - dpsif * sf * dtheta)
                        - (dphi * cf * ct - sf * dtheta) * At * (-dphi * st + dpsif)
                        + (-dphi * st + dpsif) * Ct * (dphi * cf * ct - sf * dtheta + dpsit)
                        + rf * (mf * g * st
                                - mf * ((dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * rf
                                        + (-dphi * cf * ct + sf * dtheta) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * rf))
                                + mt * g * st
                                - mt * (-(dphi * cf * ct - sf * dtheta) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * (rf + rt)
                                        + (dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw))
                        + rt * (mt * g * st
                                - mt * (-(dphi * cf * ct - sf * dtheta) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * (rf + rt)
                                        + (dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw)))

    b[2] = (-T_fric * ct
            - 2 * dphi * st * Aw * dtheta
            - dtheta * Cw * (-dphi * st + dpsiw)
            + cf * term_b3_cf_inner
            - sf * term_b3_sf_inner)

    # b[3] (MATLAB b(4))
    term_b4_sf_inner = (-Af * (dphi * sf * dpsif * ct
                              + dphi * cf * st * dtheta
                              + cf * dpsif * dtheta)
                        - (dphi * sf * ct + cf * dtheta) * Cf * (-dphi * st + dpsif)
                        + (-dphi * st + dpsif) * Bf * (dphi * sf * ct + cf * dtheta)
                        + Ct * (-dphi * sf * dpsif * ct
                                - dphi * cf * st * dtheta
                                - cf * dpsif * dtheta))
    term_b4_cf_inner = (-Bf * (dphi * cf * dpsif * ct
                              - dphi * sf * st * dtheta
                              - dpsif * sf * dtheta)
                        - (-dphi * st + dpsif) * Af * (-dphi * cf * ct + sf * dtheta)
                        + (-dphi * cf * ct + sf * dtheta) * Cf * (-dphi * st + dpsif)
                        - At * (dphi * cf * dpsif * ct
                                - dphi * sf * st * dtheta
                                - dpsif * sf * dtheta)
                        - (dphi * cf * ct - sf * dtheta) * At * (-dphi * st + dpsif)
                        + (-dphi * st + dpsif) * Ct * (dphi * cf * ct - sf * dtheta + dpsit)
                        + rf * (mf * g * st
                                - mf * ((dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * rf
                                        + (-dphi * cf * ct + sf * dtheta) * (
                                                -cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * rf))
                                + mt * g * st
                                - mt * (-(dphi * cf * ct - sf * dtheta) * (
                                        -cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * (rf + rt)
                                        + (dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw))
                        + rt * (mt * g * st
                                - mt * (-(dphi * cf * ct - sf * dtheta) * (
                                        -cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                                        + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * (rf + rt)
                                        + (dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw)))

    b[3] = (-dphi ** 2 * st * Aw * ct
            - dphi * ct * Cw * (-dphi * st + dpsiw)
            - rw * (mw * dphi * ct * (-dphi * st + dpsiw) * rw
                    - mt * g * st
                    - mw * g * st
                    + mf * ((dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw
                            + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * rf
                            + (-dphi * cf * ct + sf * dtheta) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * rf))
                    - mf * g * st
                    + mt * (-(dphi * cf * ct - sf * dtheta) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                            + (dphi * cf * dpsif * ct - dphi * sf * st * dtheta - dpsif * sf * dtheta) * (rf + rt)
                            + (dphi * sf * ct + cf * dtheta) * sf * (-dphi * st + dpsiw) * rw))
            + sf * term_b4_sf_inner
            + cf * term_b4_cf_inner)

    # b[4] (MATLAB b(5))
    term_b5_rw_sf = (mf * (-cf * dpsif * (-dphi * st + dpsiw) * rw
                          + sf * dphi * ct * dtheta * rw
                          + (dphi * sf * ct + cf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * rf)
                          - (-dphi * st + dpsif) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * rf))
                     - mf * g * cf * ct
                     - mt * g * cf * ct
                     - mt * (cf * dpsif * (-dphi * st + dpsiw) * rw
                             - sf * dphi * ct * dtheta * rw
                             + (-dphi * st + dpsif) * (-cf * (-dphi * st + dpsiw) * rw - (-dphi * st + dpsif) * (rf + rt))
                             - (dphi * sf * ct + cf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt))))
    term_b5_rw_cf = (mf * (sf * dpsif * (-dphi * st + dpsiw) * rw
                          + cf * dphi * ct * dtheta * rw
                          - (-dphi * cf * ct + sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * rf)
                          + dphi * ct * dtheta * rf
                          - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw)
                     + mt * g * sf * ct
                     + mf * g * sf * ct
                     + mt * (sf * dpsif * (-dphi * st + dpsiw) * rw
                             + cf * dphi * ct * dtheta * rw
                             - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw
                             + dphi * ct * dtheta * (rf + rt)
                             + (dphi * cf * ct - sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt))))

    b[4] = (-T_fric * st
            + 2 * Cw * dphi * ct * dtheta
            + (dphi * sf * ct + cf * dtheta) * Af * (-dphi * cf * ct + sf * dtheta)
            - rt * (-mt * g * sf * ct
                    - mt * (sf * dpsif * (-dphi * st + dpsiw) * rw
                            + cf * dphi * ct * dtheta * rw
                            - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw
                            + dphi * ct * dtheta * (rf + rt)
                            + (dphi * cf * ct - sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt))))
            - (dphi * sf * ct + cf * dtheta) * Ct * (dphi * cf * ct - sf * dtheta + dpsit)
            + At * dphi * ct * dtheta
            + rw * (2 * mw * rw * dphi * ct * dtheta
                    + sf * term_b5_rw_sf
                    + cf * term_b5_rw_cf)
            + (dphi * cf * ct - sf * dtheta) * At * (dphi * sf * ct + cf * dtheta)
            - rf * (-mt * g * sf * ct
                    - mt * (sf * dpsif * (-dphi * st + dpsiw) * rw
                            + cf * dphi * ct * dtheta * rw
                            - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw
                            + dphi * ct * dtheta * (rf + rt)
                            + (dphi * cf * ct - sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * (rf + rt)))
                    - mf * g * sf * ct
                    - mf * (sf * dpsif * (-dphi * st + dpsiw) * rw
                            + cf * dphi * ct * dtheta * rw
                            - (-dphi * cf * ct + sf * dtheta) * (dtheta * rw + (dphi * sf * ct + cf * dtheta) * rf)
                            + dphi * ct * dtheta * rf
                            - (-dphi * st + dpsif) * sf * (-dphi * st + dpsiw) * rw))
            - (-dphi * cf * ct + sf * dtheta) * Bf * (dphi * sf * ct + cf * dtheta))

    # dz(1:5) = -A\b
    dz_1to5 = -np.linalg.solve(A, b)

    # Build full dz
    dz = np.zeros(12)
    dz[0:5] = dz_1to5
    dz[5] = rw * np.cos(phi) * dpsiw
    dz[6] = rw * np.sin(phi) * dpsiw
    dz[7:12] = z[0:5]

    return dz
