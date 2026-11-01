"""N7 (level D) validation on planted populations with exact ground truth.  Development used seeds 0-2; T-N7-01 uses fresh seeds 10-14; T-N7-02 attempt 1 used seeds 10-14 (13/15 torus modules right, failed), then the bar-count rule was replaced by a shuffle-null-calibrated threshold (post hoc) and attempt 2 (null rule, seeds 20-24) FAILED worse (0 bars everywhere; a ring split into 3 modules) and was rejected; attempt 3 = fixed lifetime-ratio rule 1.5 on fresh seeds 30-34.
Thresholds fixed here before running on fresh seeds:  modules exact; spacing rel err <= 0.05; orientation err <= 5 deg; hex-template R2 >= 0.7;
connectivity: k exact, every torus module b1 == 2, random wiring b1 == 0, ring b1 == 1.  Authors: Arjun Paradkar and Reva Paradkar."""
import numpy as np
from bench.bio import gridlib as G


def test_n7_01_response_route(cell):
    res_all = []; ok = True
    for seed in range(10, 15):
        mods = [(0.45, 10 + seed, 40), (0.65, 25 + seed, 40), (0.9, 40 - seed, 40)]
        RM, T = G.planted_population(mods, arena=3.0, seed=seed, bins=72); r = G.response_route(RM, 3.0); sc = G.score_response_route(r, T)
        good = sc['pass_'] and all(e['r2'] >= 0.7 for e in sc['per_module'] if e); ok &= good; res_all.append(dict(seed=seed, k=r['n_modules'], pass_=bool(good), per_module=sc['per_module']))
    cell('T-N7-01', {'seeds': list(range(10, 15))}, 10, {'runs': res_all}, {'all five seeds pass': ok})


def test_n7_02_connectivity_route(cell):
    out = []; ok_k = ok_b = True
    for seed in range(30, 35):
        W, perm, ph = G.planted_connectivity([120, 120, 120], seed=seed); r = G.connectivity_route(W)
        out.append(dict(seed=seed, k=r['k'], b1=[m['b1'] for m in r['modules']])); ok_k &= r['k'] == 3; ok_b &= all(m['b1'] == 2 for m in r['modules'])
    rng = np.random.default_rng(30); R = rng.random((360, 360)); R = (R + R.T) / 2; rr = G.connectivity_route(R)
    p = np.random.default_rng(31).uniform(0, 1, 240); d = np.abs(p[:, None] - p[None]); d = np.minimum(d, 1 - d); ring = G.connectivity_route(np.exp(-d ** 2 / (2 * 0.08 ** 2)))
    cell('T-N7-02', {'seeds': list(range(30, 35))}, 30, {'runs': out, 'random_k': rr['k'], 'random_b1': [m['b1'] for m in rr['modules']], 'ring_b1': [m['b1'] for m in ring['modules']]},
         {'k exact on all': ok_k, 'torus b1==2 on all modules': ok_b, 'random wiring b1==0': all(m['b1'] == 0 for m in rr['modules']), 'ring b1==1': [m['b1'] for m in ring['modules']] == [1]})
