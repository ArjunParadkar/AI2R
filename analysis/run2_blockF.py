"""Run 2, Block F (SPEC2.md): the formation vineyard.  Vines = the symmetry-informed frequency circles (stated as such),
tracked by frequency label across checkpoints on the b=0 orbit probe."""
import sys, os, json, time
import numpy as np, torch
import ot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load, acts, slice_probe
from systems.z113 import P
from systems.z113_probe import psi_of
from whiteboxtopo.topology import isotypic_decomposition, persistence, circular_coordinates, circular_correlation
from whiteboxtopo.extract import jacobians

SEEDS = [('z113_clean_s0', 0, 'RE-SCORE'), ('z113_clean_s1', 1, 'RE-SCORE'), ('z113_cleanctrl_s2', 2, 'RE-SCORE'),
         ('z113_decoy_s0', 0, 'UNSEEN'), ('z113_decoy_s1', 1, 'UNSEEN')]


def circles(X, a, n_freq=6):
    out = {}
    for d in isotypic_decomposition(X, a, P, n_freq=n_freq, detect='dft', var_floor=0.0)[0]:
        Y = np.ascontiguousarray(d['proj']).astype(np.float32); res = persistence(Y); h1 = res['dgms'][1]
        rec = dict(var=float(d['var']), r=0.0, ratio=0.0, nlife=0.0, present=False, geom=(d['basis'], d['dual'], d['mean']))
        if len(h1):
            life = h1[:, 1] - h1[:, 0]; srt = np.sort(life); rec['ratio'] = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else 99.0
            rec['nlife'] = float(srt[-1] / res['median_dist'])
            cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=int(np.argmax(life)))
            rec['r'] = float(circular_correlation(cc['theta'], 2 * np.pi * d['freq'] * a / P)) if cc else 0.0
            rec['present'] = bool(rec['r'] > 0.9 and rec['ratio'] > 3.0 and rec['var'] > 0.02)      # run-1 certification rule
        out[int(d['freq'])] = rec
    return out


def w2(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    M = ot.dist(A, B, metric='sqeuclidean'); return float(np.sqrt(ot.emd2(np.ones(len(A)) / len(A), np.ones(len(B)) / len(B), M)))


def main():
    t0 = time.time(); sl = slice_probe(); a = sl['a']; out = {}
    for run, seed, status in SEEDS:
        cks, steps = checkpoints(f'runs/{run}'); hist = {h['step']: h for h in json.load(open(f'runs/{run}/history.json'))['history']}
        te = np.array([hist[int(s)]['test_acc'] for s in steps]); grok = int(np.argmax(np.diff(te))) + 1
        per = []
        for ck in cks:
            m = load(ck, seed=seed); per.append(circles(acts(m, sl['toks'], 'resid_mid'), a))
        freqs = sorted({k for c in per for k, v in c.items() if v['present']}); vines = []
        psi = psi_of(m, 'resid_mid'); Xf = acts(m, sl['toks'], 'resid_mid').astype(np.float32); Jf = jacobians(psi, Xf).astype(np.float32)
        for k in freqs:
            pres = [bool(c.get(k, {}).get('present', False)) for c in per]
            onset = next((i for i in range(len(pres)) if all(pres[i:])), None)
            nl = [c.get(k, {}).get('nlife', 0.0) for c in per]; s_c = None
            if k in per[-1]:
                E, Dl, mu = per[-1][k]['geom']; z = (Xf - mu) @ Dl; v = -z[:, 1:2] * E[:, 0] + z[:, 0:1] * E[:, 1]
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12); s_c = float((np.einsum('nod,nd->no', Jf, v) ** 2).sum(1).mean())
            vines.append(dict(freq=k, onset=onset, survives=onset is not None, peak_nlife=float(max(nl)), nlife=nl, s_c_final=s_c,
                              first_seen=int(np.argmax(pres)) if any(pres) else None))
        surv = [v for v in vines if v['survives']]; on = np.array([v['onset'] for v in surv])
        vals, cnt = np.unique(on, return_counts=True); mode = float(np.median(vals[cnt == cnt.max()])) if len(on) else None
        med = float(np.median([v['peak_nlife'] for v in surv])) if surv else None
        out[run] = dict(status=status, grok_ckpt=grok, n_ckpt=len(cks), vines=vines, n_surviving=len(surv), onset_mode=mode,
                        mode_offset_from_grok=None if mode is None else mode - grok, median_peak_nlife=med,
                        points=[[v['onset'] - grok, v['peak_nlife'] / med] for v in surv])
        print(f"[F] {run} ({status}): grok ckpt {grok}; surviving vines {len(surv)} at freqs {[v['freq'] for v in surv]} onsets {on.tolist()} "
              f"mode {mode} (offset {out[run]['mode_offset_from_grok']}); median peak nlife {med:.3f}", flush=True)
    names = [r for r, _, _ in SEEDS]; rng = np.random.default_rng(0); pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = out[names[i]], out[names[j]]
            if not A['points'] or not B['points']:
                continue
            real = w2(A['points'], B['points']); null = []
            for _ in range(200):
                Bn = [[rng.integers(0, B['n_ckpt']) - B['grok_ckpt'], p[1]] for p in B['points']]; null.append(w2(A['points'], Bn))
            pairs[f'{names[i]}|{names[j]}'] = dict(w2=real, null_p05=float(np.percentile(null, 5)), null_median=float(np.median(null)), met=bool(real < np.percentile(null, 5)))
    cnts = np.array([out[n]['n_surviving'] for n in names], float); meds = np.array([out[n]['median_peak_nlife'] for n in names], float)
    v = dict(mode_within_2={n: bool(out[n]['mode_offset_from_grok'] is not None and abs(out[n]['mode_offset_from_grok']) <= 2) for n in names},
             w2_pairs=pairs, all_pairs_met=bool(all(p['met'] for p in pairs.values())),
             vine_count_spread=float((cnts.max() - cnts.min()) / cnts.mean()), median_lifetime_spread=float((meds.max() - meds.min()) / meds.mean()))
    v['PASS'] = bool(all(v['mode_within_2'].values()) and v['all_pairs_met'] and v['vine_count_spread'] < 0.30 and v['median_lifetime_spread'] < 0.30)
    out['_verdict'] = v; print(json.dumps(v, indent=1)); print(f'wall {time.time()-t0:.0f}s')
    for n in names:
        for vv in out[n]['vines']:
            vv.pop('geom', None)
    json.dump(out, open('results/run2/blockF.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
