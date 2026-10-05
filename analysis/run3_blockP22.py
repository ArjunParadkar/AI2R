"""Run 3, P2.2 (SPEC3.md): run 1's boundary prediction, recomputed with the DISCOVERED generator and characters (nothing supplied).
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022)."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load as load_cuda
from analysis.z113_extras import rep_probe
from analysis.run3_blockZ import all_logits, D_g, P, fold
from systems.z113 import build_data
from whiteboxtopo.problems import p2_explain
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters

out = {}
for run, seed, frac in (('z113_clean_s0', 0, .3), ('z113_clean_s1', 1, .3), ('z113_cleanctrl_s2', 2, .5)):
    m = load_cuda(checkpoints(f'runs/{run}')[0][-1], seed=seed); r = automorphisms(D_g(all_logits(m.to('cpu')))); m.to('cuda')
    src = 'letter' if r['has_full_cycle'] else ('POST-HOC A-10' if r['has_full_cycle_rel'] else None)
    if src is None:
        out[run] = dict(generator=None); continue
    sig = full_cycle(r) if src == 'letter' else full_cycle(r, 'perms_rel'); t = cycle_coordinate(sig)
    ch = characters(m.W_E[:P].detach().float().cpu().numpy(), t, top=5)
    cost = np.array([sum(c['power_share'] * (1 - np.cos(2 * np.pi * c['freq'] * n / P)) for c in ch) for n in range(1, P // 2 + 1)])
    pw = np.arange(P); img = [0]
    for _ in range(P // 2):
        pw = sig[pw]; img.append(int(pw[0]))                                   # sigma^n(0)
    top3 = [fold(img[n]) for n in (np.argsort(cost)[:3] + 1)]
    ld, fit = rep_probe(m, build_data(seed, 'clean', frac=frac), 1500); ex = p2_explain(ld, fit, list(range(300)))
    dgeo = np.array([fold(e['flips_to'] - e['pred']) for e in ex])
    out[run] = dict(generator=src, generator_moves_0_to=int(sig[0]), discovered_freqs=[c['freq'] for c in ch], predicted_top3_shifts=top3,
                    frac_geodesic_in_top3=float(np.isin(dgeo, top3).mean()), chance=3 / 56, met=bool(np.isin(dgeo, top3).mean() >= 0.536 and src == 'letter'))
    print(run, out[run], flush=True)
json.dump(out, open('results/run3/blockP22.json', 'w'), indent=1)
