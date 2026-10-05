"""Run 2, D5 (EXPLORATORY, no criterion): with z hidden, do output-sensitive, low-dimensional, early H^1 classes single out the
planted shortcut?  g-geodesic Rips on a 1200-point training probe of the planted models; candidates = 30 longest-lived classes;
score = s_c / (mean local ID on the cocycle's support).  Earliness is handled by running at successive checkpoints from onset.
Reported: the rank (by score) of the class whose coordinate best tracks the hidden z, and how well it tracks it."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import load
from systems.z113 import build_data, P, NULL
from systems.z113_probe import acts, psi_of
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems.core import geodesic_distance_matrix, _coordinate_tangent
from whiteboxtopo.topology import persistence, circular_coordinates, circular_correlation
from whiteboxtopo.extract import jacobians


def main():
    out = {}
    for seed in (0, 1):
        d = build_data(seed, 'shortcut', frac=0.5); tk = d['train_toks'][np.random.default_rng(0).choice(len(d['train_toks']), 1200, replace=False)]
        z = tk[:, 2]; has = z != NULL
        for step in (500, 1000, 2500, 25000):
            m = load(f'runs/z113_shortcut_s{seed}/ckpt_{step:06d}.pt', seed=seed)
            X = acts(m, tk, 'resid_mid').astype(np.float32); J = jacobians(psi_of(m, 'resid_mid'), X).astype(np.float32)
            fit = fit_manifold(X, J, metric='g'); res = persistence(geodesic_distance_matrix(fit), distance_matrix=True); h1 = res['dgms'][1]
            life = h1[:, 1] - h1[:, 0]; life = np.where(np.isfinite(life), life, 0); rows = []
            for ci in np.argsort(-life)[:30]:
                cc = circular_coordinates(None, h1, res['cocycles'][1], D=res['D'], class_idx=int(ci))
                if cc is None:
                    continue
                v = _coordinate_tangent(fit, cc['theta']); s = float((np.einsum('nod,nd->no', J, v) ** 2).sum(1).mean())
                sup = np.unique(np.asarray(res['cocycles'][1][int(ci)])[:, :2].astype(int)); lid = float(np.nanmean(fit.d_local[sup]))
                rz = max(abs(circular_correlation(cc['theta'][has], 2 * np.pi * k * z[has] / P)) for k in range(1, 57))
                rows.append(dict(class_idx=int(ci), lifetime=float(life[ci]), s_c=s, local_id=lid, score=s / max(lid, 1e-9), r_with_hidden_z=float(rz)))
            rows.sort(key=lambda r: -r['score'])
            best = int(np.argmax([r['r_with_hidden_z'] for r in rows])) if rows else None
            out[f's{seed}_step{step}'] = dict(n_h1=int(len(h1)), n_candidates=len(rows), best_z_tracking_r=rows[best]['r_with_hidden_z'] if rows else None,
                                              rank_of_best_z_class=(best + 1) if rows else None, top5=rows[:5])
            print(f"[D5] seed {seed} step {step}: H1 {len(h1)}; best z-tracking class r={rows[best]['r_with_hidden_z']:.3f}, rank {best+1}/{len(rows)} by score", flush=True)
    json.dump(out, open('results/run2/blockD5.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
