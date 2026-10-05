"""Phase 6 H6.5 on the Gardner et al. 2022 data (prereg/PHASE_6.md).  python analysis/phase6_gardner.py rat_q|rat_r_day1|rat_s"""
import sys, json, time, os
import numpy as np
sys.path.insert(0, '.'); os.environ.setdefault('OMP_NUM_THREADS', '4')
from bench.bio import gridlib as G, gardner as D
from sklearn.metrics import adjusted_rand_score
name = sys.argv[1]; t0 = time.time(); mods, t, x, y, win = D.load(name); res = dict(file=name, window=win)
ids, RM = D.ratemaps(mods, t, x, y, win); truth = np.array([int(k[-1]) for k, _ in ids]); res['n_neurons'] = int(len(ids)); res['true_modules'] = int(len(set(truth)))
r = G.response_route(RM, 1.5, maxlag=20); lab = r['labels']; res['recovered'] = dict(n_modules=int(r['n_modules']), silhouette=float(r['silhouette']), ari=float(adjusted_rand_score(truth, lab)),
    modules=[{k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in m.items() if k in ('n', 'spacing', 'orientation', 'r2_characters')} for m in r['modules']])
# classical per-module lattice from the authors' labels (reference)
ref = []
for mk in sorted(set(truth)):
    sub = RM[truth == mk]; P = G.profiles(sub, 20); L = G.lattice_from_profile(P.mean(0).reshape(41, 41), 1.5 / 40, 20)
    ref.append(dict(module=int(mk), n=int((truth == mk).sum()), spacing=None if L is None else L['spacing'], orientation=None if L is None else L['orientation']))
res['authors_module_lattice'] = ref; res['H6.5a_pass'] = bool(r['n_modules'] == res['true_modules'] and res['recovered']['ari'] >= 0.8)
print(name, 'modules', r['n_modules'], 'truth', res['true_modules'], 'ARI', round(res['recovered']['ari'], 3), 'ref', ref, flush=True)
res['torus'] = {}
for k in sorted(mods):
    X = D.population_matrix(mods[k], win); topo = D.topology(X); null = D.shuffle_null(X, n_null=20); call = D.torus_call(topo, null)
    call['torus_called'] = bool(call['b1'] >= 2); call['n_neurons'] = int(X.shape[1]); call['h1_top5'] = topo['h1'][:5].tolist(); res['torus'][k] = call
    print(name, k, call, round(time.time() - t0), 's', flush=True); json.dump(res, open(f'results/phase6/gardner_{name}.json', 'w'), indent=1, default=float)
json.dump(res, open(f'results/phase6/gardner_{name}.json', 'w'), indent=1, default=float)
