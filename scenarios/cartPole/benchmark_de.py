import os, sys, time, numpy as np
os.environ['PYTHONWARNINGS'] = 'ignore'
_project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pilco_python.scenarios.cartPole.settings_cp import define_settings
from pilco_python.base.rollout import rollout
from pilco_python.base.trainDynModel import train_dyn_model
from pilco_python.base.propagated import propagated
from pilco_python.util.gaussian import gaussian
from pilco_python.control.congp import congp
from pilco_python.control.conCat import conCat
from pilco_python.util.gSat import gSat
from pilco_python.gp.train import train as gp_train
from pilco_python.gp.gp1d import gp1d
from pilco_python.base.value import value as pilco_value
from pilco_python.base.value_de import value_de
from pilco_python.base.analytical_variance import compute_trajectory_and_stats
from pilco_python.util.minimize import minimize


def make_ctrl():
    def ctr(p, m, s, compute_derivatives=False):
        return conCat(congp, gSat, p, m, s, compute_derivatives=compute_derivatives)
    return ctr


def deep_copy_policy(pol):
    return {
        'maxU': pol['maxU'].copy(),
        'fcn': pol['fcn'],
        'p': {k: v.copy() for k, v in pol['p'].items()},
    }


def main():
    import warnings
    warnings.filterwarnings('ignore')

    seed = 43
    n_eps = 4
    bfgs_iter = 12
    trainOpt = [30, 0]

    np.random.seed(seed)

    s = define_settings()
    plant, cost, H = s['plant'], s['cost'], s['H']
    mu0, S0 = s['mu0'], s['S0']
    dyno = s['dyno']
    plant['ctrl'] = 'zoh'
    plant['prop'] = propagated
    mu0Sim = np.zeros(len(dyno))
    S0Sim = np.diag([0.01] * len(dyno))

    n_initial = 5
    x_init = np.zeros((0, 7))
    y_init = np.zeros((0, 4))
    xx, yy, _, _ = rollout(gaussian(mu0, S0), s['policy'],
                            n_initial, plant, cost, compute_cost=False)
    x_init = np.vstack([x_init, xx])
    y_init = np.vstack([y_init, yy])

    methods = ['PILCO', 'DE-GI', 'DE-UCB']
    all_results = {}

    for mi, method in enumerate(methods):
        print(f"\n{'='*60}\n  {method}  [{mi+1}/{len(methods)}]")
        print(f"{'='*60}")

        pol = deep_copy_policy(s['policy'])
        pol['fcn'] = make_ctrl()
        dm = {'fcn': gp1d, 'train': gp_train, 'induce': np.zeros((30, 0, 1))}
        dm = train_dyn_model(x_init.copy(), y_init.copy(), dm, s['policy'],
                            plant, trainOpt)
        xd = x_init.copy()
        yd = y_init.copy()
        real_costs = []
        var_history = []
        t_start = time.time()

        for ep in range(n_eps):
            t0 = time.time()
            ep_rem = n_eps - ep
            dm = train_dyn_model(xd, yd, dm, pol, plant, trainOpt)

            bo_cfg = {
                'enabled': method != 'PILCO',
                'type': 'gi' if method == 'DE-GI' else 'ucb' if method == 'DE-UCB' else 'none',
                'seed': ep * 1000 + 777,
                'beta': 2.0,
                'sigma_y': 0.5,
                'E_remaining': ep_rem,
                'use_var_grad': (method != 'PILCO'),
                'vg_dirs': 3,
            }

            def bo_obj(p, m0, S0, dyn, pl, plt, cst, h,
                       compute_gradients=True):
                return value_de(p, m0, S0, dyn, pl, plt, cst, h,
                               bo_config=bo_cfg,
                               compute_gradients=compute_gradients)

            opt = {'length': bfgs_iter, 'MFEPLS': 8, 'verbosity': 0}
            pol['p'], fX, _, _ = minimize(
                pol['p'], bo_obj, opt,
                mu0Sim, S0Sim, dm, pol, plant, cost, H)
            best_val = float(fX[-1])

            _, _, _, var_used = compute_trajectory_and_stats(
                pol['p'], mu0Sim, S0Sim, dm, pol, plant, cost, H)
            var_history.append(var_used)
            train_mu = pilco_value(pol['p'], mu0Sim, S0Sim, dm, pol, plant,
                                  cost, H, compute_gradients=False)

            xx_new, yy_new, rc, _ = rollout(
                gaussian(mu0, S0), pol, H * 2, plant, cost, compute_cost=True)
            rc_sum = np.sum(rc)
            xd = np.vstack([xd, xx_new])
            yd = np.vstack([yd, yy_new])
            real_costs.append(rc_sum)

            upright = np.any(np.abs(xx_new[:, 3] - np.pi) < 0.5)
            tr = [f'{xx_new[:,3].min():.1f}', f'{xx_new[:,3].max():.1f}']
            print(f"  ep={ep+1} real={rc_sum:.1f} μ={float(train_mu):.2f} "
                  f"σ²={var_used:.4f} f={best_val:.3f} θ=[{tr[0]},{tr[1]}] "
                  f"{'UP!' if upright else '...'} {time.time()-t0:.0f}s")

        all_results[method] = {
            'costs': real_costs, 'vars': var_history,
            'time': time.time() - t_start,
        }
        print(f"  sum={sum(real_costs):.1f}  time={time.time()-t_start:.0f}s")

    print("\n" + "=" * 80)
    print("  RESULTS: PILCO vs DE (analytical variance + corrected gradient)")
    print("=" * 80)
    for m in methods:
        cs = all_results[m]['costs']
        vs = [f'{v:.4f}' for v in all_results[m]['vars']]
        print(f"  {m:10s} costs={[f'{c:.1f}' for c in cs]}  "
              f"sum={sum(cs):.1f}  vars={vs}")

    pilco_cs = all_results['PILCO']['costs']
    print("-" * 80)
    for m in methods:
        if m == 'PILCO':
            continue
        cm = sum(all_results[m]['costs'])
        cp = sum(pilco_cs)
        pct = (cm - cp) / max(abs(cp), 1) * 100
        sgn = '+' if pct > 0 else ''
        w = 'BETTER!' if pct < 0 else 'worse'
        print(f"  {m:10s} vs PILCO: {cm:.1f} vs {cp:.1f} "
              f"({sgn}{pct:.1f}%) — {w}")
    print("=" * 80)


if __name__ == '__main__':
    main()
