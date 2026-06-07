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
from pilco_python.base.cumulative_cost import distributional_cost_stats
from pilco_python.base.directed_explore import ucb, gittins_index
from pilco_python.util.minimize import minimize


def make_ctrl():
    def ctr(p, m, s, compute_derivatives=False):
        return conCat(congp, gSat, p, m, s, compute_derivatives=compute_derivatives)
    return ctr


def main():
    import warnings
    warnings.filterwarnings('ignore')

    seed, n_eps = 42, 4
    n_mc, n_cand, pert_scale, bfgs_iter = 40, 25, 0.15, 20

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    s = define_settings()
    plant, cost, H = s['plant'], s['cost'], s['H']
    mu0, S0 = s['mu0'], s['S0']
    dyno = s['dyno']
    plant['ctrl'] = 'zoh'
    plant['prop'] = propagated
    mu0Sim = np.zeros(len(dyno))
    S0Sim = np.diag([0.01] * len(dyno))

    # Single random rollout for initial data
    x_data = np.zeros((0, 7))
    y_data = np.zeros((0, 4))
    start = gaussian(mu0, S0)
    xx, yy, _, _ = rollout(start, s['policy'], 10, plant, cost, compute_cost=False)
    x_data = np.vstack([x_data, xx])
    y_data = np.vstack([y_data, yy])

    trainOpt = [100, 100]

    results = {}
    for method in ['PILCO', 'DE-UCB', 'DE-GI']:
        print(f"\n{'='*50}\n  {method}\n{'='*50}")
        pol = {'maxU': s['policy']['maxU'].copy(), 'fcn': make_ctrl(),
               'p': {k: v.copy() for k, v in s['policy']['p'].items()}}
        dm = {'fcn': gp1d, 'train': gp_train, 'induce': np.zeros((30, 0, 1))}
        dm = train_dyn_model(x_data, y_data, dm, s['policy'], plant, trainOpt)
        xd, yd = x_data.copy(), y_data.copy()

        costs, vars_arr = [], []
        best_mu, best_var = 0.0, 0.0
        t0_all = time.time()

        for ep in range(n_eps):
            t0 = time.time()
            ep_rem = n_eps - ep
            dm = train_dyn_model(xd, yd, dm, pol, plant, trainOpt)

            opt = {'length': bfgs_iter, 'MFEPLS': 10, 'verbosity': 0}
            pol['p'], fX, _, _ = minimize(
                pol['p'], pilco_value, opt,
                mu0Sim, S0Sim, dm, pol, plant, cost, H)
            best_mu = float(fX[-1])

            if method != 'PILCO':
                best_bo = float('inf')
                best_pol = pol
                for c in range(n_cand):
                    cand = {'maxU': pol['maxU'].copy(), 'fcn': pol['fcn'],
                            'p': {k: v.copy() for k, v in pol['p'].items()}}
                    for k in cand['p']:
                        if isinstance(cand['p'][k], np.ndarray):
                            cand['p'][k] += pert_scale * rng.standard_normal(cand['p'][k].shape)
                    mu_c, var_c, _ = distributional_cost_stats(
                        cand, plant, dm, cost, mu0Sim, S0Sim, H,
                        n_samples=n_mc, seed=ep*1000+c)
                    sigma_c = np.sqrt(var_c)
                    bo = (ucb(mu_c, sigma_c, beta=1.5) if method=='DE-UCB' else
                          gittins_index(mu_c, sigma_c, sigma_y=0.5, E_remaining=ep_rem))
                    if bo < best_bo:
                        best_bo, best_pol, best_mu, best_var = bo, cand, mu_c, var_c
                pol = best_pol
            else:
                _, var_c, _ = distributional_cost_stats(
                    pol, plant, dm, cost, mu0Sim, S0Sim, H,
                    n_samples=n_mc, seed=ep*9999)
                best_var = var_c

            vars_arr.append(best_var)
            xx_new, yy_new, rc, _ = rollout(
                gaussian(mu0, S0), pol, H*2, plant, cost, compute_cost=True)
            rc_sum = np.sum(rc)
            xd = np.vstack([xd, xx_new])
            yd = np.vstack([yd, yy_new])
            costs.append(rc_sum)
            upright = np.any(np.abs(xx_new[:, 3] - np.pi) < 0.5)
            tr = [f'{xx_new[:,3].min():.1f}', f'{xx_new[:,3].max():.1f}']
            print(f"  ep={ep+1}  cost={rc_sum:.1f}  μ={best_mu:.2f}  "
                  f"σ²={best_var:.4f}  θ=[{tr[0]},{tr[1]}]  "
                  f"{'UP!'if upright else'...'}  {time.time()-t0:.0f}s")

        results[method] = {'costs': costs, 'vars': vars_arr,
                          'time': time.time()-t0_all}

    print("\n" + "=" * 70)
    print("  RESULTS: PILCO vs Directed Exploration on CartPole")
    print("  (sparse initial data, limited optimization)")
    print("=" * 70)
    for m in ['PILCO', 'DE-UCB', 'DE-GI']:
        r = results[m]
        cs = r['costs']
        row = f"{m:<12} " + " ".join(f"{c:>7.1f}" for c in cs)
        print(f"{row}  sum={sum(cs):.1f}  t={r['time']:.0f}s")

    print("-" * 70)
    pilco_cs = results['PILCO']['costs']
    for m in ['DE-UCB', 'DE-GI']:
        cm = sum(results[m]['costs'])
        cp = sum(pilco_cs)
        pct = (cm - cp) / abs(cp) * 100
        w = 'BETTER!' if pct < 0 else 'worse'
        print(f"  {m}: {cm:.1f} vs PILCO {cp:.1f} ({pct:+.1f}%) — {w}")

    print("\n  Variance (model uncertainty) over episodes:")
    for m in ['PILCO', 'DE-UCB', 'DE-GI']:
        vs = [f'{v:.4f}' for v in results[m]['vars']]
        print(f"  {m}: {vs}")
    print("=" * 70)


if __name__ == '__main__':
    main()
