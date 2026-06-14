import os, sys, time, numpy as np
os.environ['PYTHONWARNINGS'] = 'ignore'
_p = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if _p not in sys.path: sys.path.insert(0, _p)

from pilco_python.scenarios.cartDoublePendulum.settings_cdp import (
    dt, angi, poli, dyno, plant as _plant, cost as _cost, dynmodel as _dynmodel,
)
from pilco_python.scenarios.cartDoublePendulum.dynamics_cdp import dynamics_cdp
from pilco_python.scenarios.cartDoublePendulum.loss_cdp import loss_cdp
from pilco_python.base.rollout import rollout
from pilco_python.base.trainDynModel import train_dyn_model
from pilco_python.base.propagated import propagated
from pilco_python.util.gaussian import gaussian
from pilco_python.control.congp import congp; from pilco_python.control.conCat import conCat
from pilco_python.util.gSat import gSat
from pilco_python.gp.train import train as gp_train; from pilco_python.gp.gp1d import gp1d
from pilco_python.base.value import value as pilco_value
from pilco_python.base.value_de import value_de
from pilco_python.base.analytical_variance import compute_trajectory_and_stats
from pilco_python.util.minimize import minimize
from pilco_python.util.gTrig import gTrig


def make_ctrl():
    def ctr(p, m, s, compute_derivatives=False):
        return conCat(congp, gSat, p, m, s, compute_derivatives=compute_derivatives)
    return ctr


def main():
    import warnings; warnings.filterwarnings('ignore')
    seed = 42; np.random.seed(seed)

    # Paper-exact settings (from settings_cdp.py)
    nc = 200; H = 100; induce = 400
    n_eps = 20; bfgs_iter = 30; vg_dirs = 2
    trainOpt = [300, 500]
    dt = 0.05; T = 5.0

    plant = dict(_plant)
    plant['dynamics'] = dynamics_cdp; plant['prop'] = propagated
    cost = dict(_cost); cost['fcn'] = loss_cdp

    mu0 = np.array([0, 0, 0, 0, np.pi, np.pi])
    S0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01])**2

    mm, ss, cc = gTrig(mu0, S0, angi, compute_derivatives=False)
    mm_full = np.concatenate([mu0, mm]); cc_full = S0 @ cc
    ss_full = np.vstack([np.hstack([S0, cc_full]), np.hstack([cc_full.T, ss])])
    mm_p = mm_full[poli]; ss_p = ss_full[np.ix_(poli, poli)]
    inputs = gaussian(mm_p, ss_p, nc).T
    targets = 0.1 * np.random.randn(nc, 1)
    hyp = np.log(np.concatenate([np.ones(len(poli)), [1.0], [0.01]]))
    pb = {'maxU': np.array([20.0]), 'fcn': None,
          'p': {'inputs': inputs, 'targets': targets, 'hyp': hyp}}

    mu0Sim = np.zeros(len(dyno)); S0Sim = np.diag(np.ones(len(dyno)) * 0.01**2)

    # J=1 initial random trajectory of length H (paper setting)
    x_init = np.zeros((0, 10 + 1)); y_init = np.zeros((0, 6))
    xx, yy, _, _ = rollout(gaussian(mu0, S0), {'maxU': np.array([20.0]), 'fcn': None},
                            H, plant, cost, compute_cost=False)
    x_init = np.vstack([x_init, xx]); y_init = np.vstack([y_init, yy])
    n_init = len(xx)

    print(f"Config: nc={nc} H={H} induce={induce} eps={n_eps} bfgs={bfgs_iter}")
    print(f"  trainOpt={trainOpt} init={n_init} steps vg_dirs={vg_dirs}")
    print(f"  Cost max = {H} (per episode)")
    print(f"  Started at {time.strftime('%H:%M:%S')}")

    methods = ['PILCO', 'DE-GI']
    results = {}; t_global = time.time(); total_ep = 0

    for mi, method in enumerate(methods):
        print(f"\n{'='*60}\n  {method}  [{mi+1}/{len(methods)}]")
        pol = {'maxU': pb['maxU'].copy(), 'fcn': make_ctrl(),
               'p': {k: v.copy() for k, v in pb['p'].items()}}
        dm = {'fcn': gp1d, 'train': gp_train, 'induce': np.zeros((induce, 0, 1))}
        t0 = time.time()
        dm = train_dyn_model(x_init.copy(), y_init.copy(), dm,
                            {'maxU': np.array([20.0]), 'fcn': None}, plant, trainOpt)
        print(f"  Init GP: {time.time()-t0:.0f}s")
        xd, yd = x_init.copy(), y_init.copy()
        cs, vh = [], []; _, _, _, v0 = compute_trajectory_and_stats(
            pol['p'], mu0Sim, S0Sim, dm, pol, plant, cost, H)
        print(f"  Init sig2={v0:.2f}")

        for ep in range(n_eps):
            t_ep = time.time(); er = n_eps - ep; total_ep += 1

            t0 = time.time()
            dm = train_dyn_model(xd, yd, dm, pol, plant, trainOpt)
            gp_t = time.time() - t0

            if method == 'PILCO':
                def bo(p_, m0, S0, dyn, pl, plt, cst, h, cg=True):
                    return pilco_value(p_, m0, S0, dyn, pl, plt, cst, h, compute_gradients=cg)
            else:
                cfg = {'enabled': True, 'type': 'gi', 'beta': 2.0, 'sigma_y': 0.5,
                       'E_remaining': er, 'vg_dirs': vg_dirs, 'seed': ep * 1000 + seed}
                def bo(p_, m0, S0, dyn, pl, plt, cst, h, cg=True):
                    return value_de(p_, m0, S0, dyn, pl, plt, cst, h,
                                   bo_config=cfg, compute_gradients=cg)

            t0 = time.time()
            opt = {'length': bfgs_iter, 'MFEPLS': 8, 'verbosity': 0}
            pol['p'], fX, _, _ = minimize(pol['p'], bo, opt,
                                          mu0Sim, S0Sim, dm, pol, plant, cost, H)
            bfs_t = time.time() - t0

            t0 = time.time()
            _, _, _, var_u = compute_trajectory_and_stats(
                pol['p'], mu0Sim, S0Sim, dm, pol, plant, cost, H)
            vh.append(var_u)
            mu_t = pilco_value(pol['p'], mu0Sim, S0Sim, dm, pol, plant, cost, H,
                              compute_gradients=False)

            xx_n, yy_n, rc, _ = rollout(gaussian(mu0, S0), pol, H,
                                        plant, cost, compute_cost=True)
            rc_s = np.sum(rc); xd = np.vstack([xd, xx_n]); yd = np.vstack([yd, yy_n])
            cs.append(rc_s)
            up = np.any(np.abs(xx_n[:, 4] - np.pi) < 0.5) and np.any(np.abs(xx_n[:, 5] - np.pi) < 0.5)
            ep_t = time.time() - t_ep
            print(f"  ep{ep+1}/{n_eps} real={rc_s:.0f} mu={float(mu_t):.0f} "
                  f"sig2={var_u:.1f} {'UP!' if up else '...'} "
                  f"gp={gp_t:.0f}s bfgs={bfs_t:.0f}s tot={ep_t:.0f}s "
                  f"[{time.strftime('%H:%M:%S')}]")

        results[method] = {'costs': cs, 'vars': vh}
        print(f"  {method} sum={sum(cs):.0f}")

    print(f"\n{'='*70}")
    print(f"  PILCO vs DE-GI on Cart Double-Pendulum")
    print(f"  nc={nc} H={H} eps={n_eps} bfgs={bfgs_iter} train={trainOpt}")
    print(f"{'='*70}")
    for m in methods:
        c = results[m]['costs']
        print(f"  {m:8s} costs={[f'{x:.0f}' for x in c]} sum={sum(c):.0f}")
    pc = sum(results['PILCO']['costs'])
    if 'DE-GI' in results:
        cm = sum(results['DE-GI']['costs']); cp = sum(pc)
        pct = (cm - cp) / max(abs(cp), 1) * 100
        print(f"  DE-GI vs PILCO: {cm:.0f} vs {cp:.0f} ({pct:+.1f}%) {'BETTER!' if pct < 0 else 'worse'}")
    print(f"  Total: {time.time()-t_global:.0f}s [{time.strftime('%H:%M:%S')}]")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
