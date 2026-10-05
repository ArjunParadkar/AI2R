"""Run 2, Block D1/D2 (SPEC2.md + AMENDMENTS A-1, A-2): Mapper alignment between the output lens and the nuisance lens,
on the decorrelated probe, across checkpoints; applied to the regenerated re-encoded model where the H^1 detector goes silent."""
import sys, os, re, json, glob, time
import numpy as np, torch
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, build_data, P, NULL, EQ
from systems.z113_probe import acts

N_ARCS, N_PROBE = 12, 2000


def probe(seed, mode, frac):
    d = build_data(seed, mode, frac=frac); rng = np.random.default_rng(0)
    te = rng.choice(len(d['test_toks']), size=min(N_PROBE, len(d['test_toks'])), replace=False)
    z = rng.integers(0, P + 1, size=len(te)); z = np.where(z == P, NULL, z)
    tk = d['test_toks'][te].copy(); tk[:, 2] = z
    return tk, z, d['test_y'][te]


def mapper_nodes(X, lens, thr):
    """node = (first cover element containing the point, single-linkage component inside that element)"""
    centres = np.arange(N_ARCS) * P / N_ARCS; half = P / N_ARCS          # width 2P/12 -> half-width P/12
    elem = np.full(len(X), N_ARCS, int)                                   # NULL lens value: its own element
    ok = lens != NULL
    dist = np.abs(((lens[ok, None] - centres[None]) + P / 2) % P - P / 2)
    elem[ok] = np.argmax(dist <= half, axis=1)
    node = np.zeros(len(X), int); nxt = 0
    for e in np.unique(elem):
        ii = np.where(elem == e)[0]
        if len(ii) == 1:
            node[ii] = nxt; nxt += 1; continue
        G = radius_neighbors_graph(X[ii], thr, mode='connectivity', include_self=False)
        nc, lab = connected_components(G, directed=False); node[ii] = nxt + lab; nxt += nc
    return node


def alignment(model, tk, z):
    with torch.no_grad():
        yhat = model(torch.as_tensor(tk, device='cuda')).argmax(-1).cpu().numpy()
    X = acts(model, tk, 'resid_mid').astype(np.float32)
    thr = float(np.median(NearestNeighbors(n_neighbors=6).fit(X).kneighbors(X)[0][:, 5]))
    ny, nz = mapper_nodes(X, yhat, thr), mapper_nodes(X, z, thr)
    present = z != NULL
    return dict(A=float(NMI(ny, nz)), A_ami=float(AMI(ny, nz)), n_nodes_y=int(ny.max() + 1), n_nodes_z=int(nz.max() + 1),
                direct_nmi=float(NMI(yhat, z)), direct_ami=float(AMI(yhat, z)),
                frac_pred_equals_hint=float((yhat[present] == z[present]).mean())), yhat


def scan(run, seed, mode, frac, every=1):
    tk, z, y = probe(seed, mode, frac); rows = []
    cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[::every]
    for ck in cks:
        m = Z113Transformer(seed=seed).cuda(); m.load_state_dict(torch.load(ck, map_location='cuda')); m.eval()
        r, yhat = alignment(m, tk, z); r['step'] = int(re.search(r'ckpt_(\d+)', ck)[1]); r['dec_acc'] = float((yhat == y).mean()); rows.append(r)
    return rows


LOG = re.compile(r'\]\s+(\d+) test [\d.]+ dec [\d.]+ worst [\d.]+ reliance ([-\d.]+) spurious (\d+)')


def h1_log(path):
    return {int(m[1]): dict(reliance=float(m[2]), h1_spurious=int(m[3])) for m in map(LOG.search, open(path)) if m}


def main():
    t0 = time.time(); out = {}
    runs = [('z113_clean_s0', 0, 'clean', .3), ('z113_clean_s1', 1, 'clean', .3), ('z113_cleanctrl_s2', 2, 'clean', .5),
            ('z113_decoy_s0', 0, 'decoy', .5), ('z113_decoy_s1', 1, 'decoy', .5),
            ('z113_shortcut_s0', 0, 'shortcut', .5), ('z113_shortcut_s1', 1, 'shortcut', .5),
            ('r2_reencoded_s0', 0, 'shortcut', .5), ('r2_reencoded_s1', 1, 'shortcut', .5)]
    for run, seed, mode, frac in runs:
        if not glob.glob(f'runs/{run}/ckpt_*.pt'):
            print('missing', run); continue
        out[run] = scan(run, seed, mode, frac)
        A = np.array([r['A'] for r in out[run]]); print(f"[D1] {run:20s} n={len(A)} A mean {A.mean():.4f} sd {A.std():.4f} min {A.min():.4f} max {A.max():.4f} | "
              f"AMI max {max(r['A_ami'] for r in out[run]):.3f} | direct NMI max {max(r['direct_nmi'] for r in out[run]):.3f} | nodes ~{out[run][-1]['n_nodes_y']}/{out[run][-1]['n_nodes_z']}", flush=True)
    cal = np.array([r['A'] for k in ('z113_clean_s0', 'z113_clean_s1') for r in out[k]])
    thr = float(cal.mean() + 3 * cal.std()); v = dict(threshold=thr, cal_mean=float(cal.mean()), cal_sd=float(cal.std()))
    cal_d = np.array([r['direct_nmi'] for k in ('z113_clean_s0', 'z113_clean_s1') for r in out[k]]); thr_d = float(cal_d.mean() + 3 * cal_d.std()); v['threshold_direct'] = thr_d
    for k, rows in out.items():
        for r in rows:
            r['fires'] = bool(r['A'] > thr); r['fires_direct'] = bool(r['direct_nmi'] > thr_d)
    v['null_firings'] = {k: int(sum(r['fires'] for r in out[k])) for k in ('z113_cleanctrl_s2', 'z113_decoy_s0', 'z113_decoy_s1') if k in out}
    v['null_firings_direct'] = {k: int(sum(r['fires_direct'] for r in out[k])) for k in ('z113_cleanctrl_s2', 'z113_decoy_s0', 'z113_decoy_s1') if k in out}
    v['shortcut_firings'] = {k: f"{sum(r['fires'] for r in out[k])}/{len(out[k])}" for k in ('z113_shortcut_s0', 'z113_shortcut_s1') if k in out}
    for s in (0, 1):
        k = f'r2_reencoded_s{s}'
        if k not in out:
            continue
        lg = h1_log(f'logs/run2_reencoded_s{s}.log')
        for r in out[k]:
            r.update(lg.get(r['step'], {}))
        win = [r for r in out[k] if r.get('h1_spurious') == 0 and r.get('reliance', 0) > 0.9]
        v[k] = dict(evasion_window_steps=[r['step'] for r in win], evasion_reproduced=bool(len(win) >= 1),
                    A_in_window=[round(r['A'], 4) for r in win], fires_in_window=[r['fires'] for r in win],
                    direct_fires_in_window=[r['fires_direct'] for r in win], ami_in_window=[round(r['A_ami'], 4) for r in win],
                    D2_MET=bool(len(win) >= 1 and all(r['fires'] for r in win)))
    out['_verdict'] = v; print(json.dumps(v, indent=1)); print(f'wall {time.time()-t0:.0f}s')
    json.dump(out, open('results/run2/blockD12.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
