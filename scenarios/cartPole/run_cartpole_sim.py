"""
run_cartpole_sim.py - PILCO cart-pole training + rollout animation GIF.

Faithful translation of MATLAB cartPole_learn.m PILCO pipeline:
1. Initial rollout with random actions (no controller)
2. Train GP dynamics model on accumulated data
3. Policy search via minimize (BFGS, matching MATLAB)
4. Controlled rollout with trained policy, accumulate data
5. Repeat for N iterations
6. Generate animation GIF
"""

import os, sys, time
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pilco_python.scenarios.cartPole.settings_cp import define_settings, get_initial_rollout_data
from pilco_python.base.rollout import rollout
from pilco_python.base.trainDynModel import train_dyn_model
from pilco_python.base.learnPolicy import learn_policy
from pilco_python.base.applyController import applyController
from pilco_python.base.propagated import propagated
from pilco_python.util.gaussian import gaussian
from pilco_python.control.congp import congp
from pilco_python.control.conCat import conCat
from pilco_python.util.gSat import gSat
from pilco_python.gp.train import train as gp_train
from pilco_python.gp.gp1d import gp1d


def make_controller():
    """Create the PILCO controller: conCat(congp, gSat)."""
    def controller(pol, m, s, compute_derivatives=False):
        return conCat(congp, gSat, pol, m, s, compute_derivatives=compute_derivatives)
    return controller


def generate_gif(xx, latent, dt, filename='cartpole_trained.gif', max_frames=120):
    """Generate an animated GIF of the cart-pole rollout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    l_pend = 0.6
    xmin, xmax = -3, 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def animate_step(frame_idx):
        ax1.clear()
        ax2.clear()

        step = min(frame_idx, len(xx) - 1)
        x_pos = xx[step, 0]
        theta_val = xx[step, 3]

        # Cart-pole drawing
        cart_w, cart_h = 0.3, 0.1
        cart_rect = np.array([
            [x_pos + cart_w, cart_h], [x_pos + cart_w, -cart_h],
            [x_pos - cart_w, -cart_h], [x_pos - cart_w, cart_h],
            [x_pos + cart_w, cart_h],
        ])
        pend_x = [x_pos, x_pos + 2 * l_pend * np.sin(theta_val)]
        pend_y = [0, -np.cos(theta_val) * 2 * l_pend]

        ax1.fill(cart_rect[:, 0], cart_rect[:, 1], 'k', edgecolor='k')
        ax1.plot(pend_x, pend_y, 'r', linewidth=4)
        ax1.plot(x_pos, 0, 'y.', markersize=20)
        ax1.plot(pend_x[1], pend_y[1], 'y.', markersize=20)
        ax1.plot([xmin, xmax], [-cart_h - 0.03, -cart_h - 0.03], 'k', linewidth=2)
        ax1.plot(0, 2 * l_pend, 'k+', markersize=20, linewidth=2)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        upright = abs(theta_val - np.pi) < 0.5
        status = 'BALANCED!' if upright else 'swinging'
        ax1.set_title(f'Cart-Pole PILCO (t={step * dt:.1f}s) [{status}]', fontsize=12)

        # Phase space
        trail = min(step, 200)
        start_idx = max(0, step - trail)
        ax2.plot(xx[start_idx:step + 1, 0], xx[start_idx:step + 1, 3],
                 'b-', alpha=0.4, linewidth=0.5)
        ax2.plot(xx[step, 0], xx[step, 3], 'ro', markersize=8)
        ax2.axhline(y=np.pi, color='g', linestyle='--', alpha=0.3, label='target (upright)')
        ax2.axhline(y=-np.pi, color='g', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Cart Position x [m]')
        ax2.set_ylabel('Pendulum Angle [rad]')
        ax2.set_title('Phase Space (x vs theta)')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(xmin, xmax)

        return ax1, ax2

    total_frames = min(len(xx), max_frames)
    skip = max(1, len(xx) // total_frames)
    frames_list = list(range(0, len(xx), skip))

    print(f"  Rendering {len(frames_list)} animation frames...")
    ani = FuncAnimation(fig, animate_step, frames=frames_list,
                       interval=dt * 1000 * skip, blit=False)

    gif_path = os.path.join(os.path.dirname(__file__), filename)
    fps = min(10, int(1.0 / dt / max(1, skip)))
    writer = PillowWriter(fps=fps)
    ani.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"  GIF saved: {gif_path} ({os.path.getsize(gif_path)} bytes)")
    return gif_path


def main(num_iterations=3, generate_gif_flag=True):
    """Run PILCO cart-pole training and generate animation GIF.

    Matches MATLAB cartPole_learn.m pipeline:
    - Phase 1: random rollout (no controller, random actions)
    - Phase 2: train GP dynamics model
    - Phase 3-N: policy search + controlled rollout + re-train
    """
    print("=" * 60)
    print("  PILCO Cart-Pole Swingup: Training + Animation")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────
    settings = define_settings()
    mu0 = settings['mu0']
    S0 = settings['S0']
    plant = settings['plant']
    cost = settings['cost']
    H = settings['H']
    dt = settings['dt']
    dyno = settings['dyno']
    trainOpt = settings['trainOpt']

    plant['ctrl'] = 'zoh'
    plant['prop'] = propagated

    # Do NOT set controller yet - initial rollout uses random actions like MATLAB
    # MATLAB: rollout(gaussian(mu0,S0), struct('maxU',policy.maxU), ...)
    # struct('maxU',policy.maxU) has no 'fcn' field -> random actions
    policy = settings['policy']

    dynmodel = settings['dynmodel']
    dynmodel['fcn'] = gp1d
    dynmodel['train'] = gp_train
    dynmodel['induce'] = np.zeros((50, 0, 1))

    print(f"\n  Config: dt={dt}s, H={H}, iterations={num_iterations}")
    print(f"  Control dim: {len(policy['maxU'])}, Policy input dim: {len(plant['poli'])}")

    # ── Phase 1: Initial random rollout (NO controller) ─────
    print("\n── Phase 1: Initial random rollout ──")
    # Generate initial rollout with seed 1 (matching MATLAB rand('seed',1); randn('seed',1))
    xx_mat, yy_mat = get_initial_rollout_data(plant, policy, cost, H, mu0, S0)
    x_data = xx_mat.copy()   # full 7-column augmented state
    y_data = yy_mat.copy()   # 4-column target differences
    print(f"  Loaded {len(xx_mat)} steps of random rollout data (seed=1)")

    # ── Phase 2: Initial GP training ────────────────────────
    print("\n── Phase 2: Train GP dynamics model ──")
    t0 = time.time()
    dynmodel = train_dyn_model(x_data, y_data, dynmodel, policy, plant, trainOpt)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Set controller before policy search ─────────────────
    # MATLAB: policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s)
    policy['fcn'] = make_controller()

    # ── Phase 3-N: PILCO iterations ─────────────────────────
    print(f"\n── Phase 3: PILCO iterations ({num_iterations} total) ──")
    mu0Sim = np.zeros(len(dyno))
    S0Sim = np.diag([0.01] * len(dyno))

    for j in range(num_iterations):
        print(f"\n  >>> Iteration {j + 1}/{num_iterations} <<<")
        t_start = time.time()

        # Policy search
        policy, fX3, M_j, Sigma_j, fm_j, fs_j = learn_policy(
            mu0Sim, S0Sim, dynmodel, policy, plant, cost, H,
            plotting={'verbosity': 0})
        print(f"  Policy optimized: fX_last={fX3[-1]:.3f}")

        # Controlled rollout with trained policy
        start_state = gaussian(mu0, S0)
        xx_new, yy_new, _, latent_new = rollout(
            start_state, policy, H * 2, plant, cost, compute_cost=False)
        x_data = np.vstack([x_data, xx_new])
        y_data = np.vstack([y_data, yy_new])

        # Check rollout quality
        theta_range = [xx_new[:, 3].min(), xx_new[:, 3].max()]
        near_upright = np.any(np.abs(xx_new[:, 3] - np.pi) < 0.5)
        print(f"  Rollout: theta=[{theta_range[0]:.1f}, {theta_range[1]:.1f}] "
              f"upright={near_upright} samples={len(xx_new)}")

        # Re-train GP on accumulated data
        dynmodel = train_dyn_model(x_data, y_data, dynmodel, policy, plant, trainOpt)
        print(f"  Time: {time.time() - t_start:.1f}s  "
              f"total_data: {len(x_data)} steps")

        if near_upright:
            print(f"\n  *** SWING-UP ACHIEVED at iteration {j + 1}! ***")
            break

    # ── Phase 4: Generate animation GIF ─────────────────────
    if generate_gif_flag:
        print(f"\n── Phase 4: Generate animation ──")
        np.random.seed(123)
        start_state = gaussian(mu0, S0)
        xx_final, _, _, latent_final = rollout(
            start_state, policy, H * 3, plant, cost, compute_cost=False)
        xx_final = xx_final[:, :len(dyno)]
        print(f"  Final rollout: {len(xx_final)} steps")
        print(f"  x range: [{xx_final[:, 0].min():.2f}, {xx_final[:, 0].max():.2f}]")
        print(f"  theta range: [{xx_final[:, 3].min():.2f}, {xx_final[:, 3].max():.2f}]")

        gif_path = generate_gif(xx_final, latent_final, dt, 'cartpole_trained.gif')
    else:
        gif_path = None

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Complete!")
    print(f"  Training data: {len(x_data)} points across {num_iterations} iterations")
    print(f"  Final policy cost: {fX3[-1]:.1f}")
    if gif_path:
        print(f"  GIF: {gif_path}")
    print("=" * 60)

    return policy, dynmodel, plant, cost, gif_path


if __name__ == '__main__':
    main(num_iterations=5, generate_gif_flag=True)
