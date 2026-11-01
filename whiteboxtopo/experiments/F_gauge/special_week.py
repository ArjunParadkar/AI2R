"""F8 (renaming samples + identity), F9 (anchor), F10 (audit table) — SPEC §F8–F10.
F8/F9 on every converged specimen; F10 on seeds 0-4 of every cell (AMENDMENTS #3, compute budget).
Writes results/F8_identity.csv, results/F9_anchor.csv, results/audit_table.csv. Resumable."""
import os, sys, csv, time, argparse
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from flib.models import MLP
from flib.functions import truth_table
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import check_invariance, weight_product
from whiteboxtopo.core.persistence import compute_diagrams
from whiteboxtopo.topology import standardized_cloud

R = os.path.join(os.path.dirname(__file__), "results")
N_REP, N_PROBE = 100, 256


def _probe(n_in, seed=0):
    return np.random.default_rng(seed).uniform(-1, 2, size=(N_PROBE, n_in))


def ph_summary(cloud):
    dg = compute_diagrams(cloud, maxdim=1, backend="ripser", thresh_percentile=100.0)
    out = []
    for k in (0, 1):
        D = np.asarray(dg.dgms[k]).reshape(-1, 2); fin = np.isfinite(D[:, 1]); life = D[fin, 1] - D[fin, 0]
        p = life / life.sum() if life.sum() > 0 else np.zeros(0)
        out += [len(D), float(np.sum(life ** 2)), float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0]
    # beta1 at the median pairwise distance
    D1 = np.asarray(dg.dgms[1]).reshape(-1, 2)
    from scipy.spatial.distance import pdist
    med = float(np.median(pdist(cloud))) if len(cloud) > 1 else 0.0
    out.append(int(((D1[:, 0] <= med) & (D1[:, 1] > med)).sum()) if len(D1) else 0)
    return np.array(out, dtype=np.float64)   # [n0, tp0, pe0, n1, tp1, pe1, beta1@median]


def _acts(m, P):
    return m.hidden_acts(P)


def _gradxinput(m, P):
    """Per hidden unit: mean over probe of a_u * d logit / d a_u (single-output MLP)."""
    from whiteboxtopo.invariants import as_relu_mlp
    m = as_relu_mlp(m)
    masks, acts, out = m.forward(P)
    d = m.depth; g = np.ones((P.shape[0], 1))                        # d logit / d out
    grads = [None] * d
    for l in range(d, 0, -1):
        g = g @ m.Ws[l].T * masks[l - 1]                            # d logit / d pre_{l-1}
        grads[l - 1] = g
    return np.concatenate([(acts[l] * grads[l]).mean(0) for l in range(d)])


def _hungarian_cols(A, B):
    """Permute the columns of A to best match the columns of B (unit-normalised columns)."""
    from scipy.optimize import linear_sum_assignment
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-300); Bn = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-300)
    cost = -(An.T @ Bn)
    ri, ci = linear_sum_assignment(cost)
    perm = np.empty(A.shape[1], dtype=int); perm[ci] = ri
    return A[:, perm]


def _procrustes_cols(A, B):
    """Orthogonal R minimising ||A R - B||_F; returns A R."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return A @ (U @ Vt)


def metrics_for(m, X, P):
    """Dict name -> metric_fn(model). Both the truth table X and the probe cloud P are used."""
    d = m.depth
    def loop_ratios(mm):
        out = []
        for l in range(d):
            W = mm.Ws[l]
            r, c = W.shape
            for a in range(r):
                for a2 in range(a + 1, r):
                    for b in range(c):
                        for b2 in range(b + 1, c):
                            num = W[a, b] * W[a2, b2] - W[a, b2] * W[a2, b]
                            den = abs(W[a, b] * W[a2, b2]) + abs(W[a, b2] * W[a2, b])
                            out.append(num / den if den > 0 else 0.0)
        return np.sort(np.abs(out))
    def top_direction(mm, l, k=1):
        A = mm.hidden_acts(P)[l]; A = A - A.mean(0)
        V = np.linalg.svd(A, full_matrices=False)[2][:k]
        V = V * np.sign(V[np.arange(k), np.argmax(np.abs(V), 1)])[:, None]
        return V[0][:, None] if k == 1 else V
    M = {
        "weight_product": lambda mm: weight_product(mm),
        "ph_raw_probe": lambda mm: np.concatenate([ph_summary(a) for a in mm.hidden_acts(P)]),
        "ph_raw_truthtable": lambda mm: np.concatenate([ph_summary(a) for a in mm.hidden_acts(X)]),
        "ph_std_probe": lambda mm: np.concatenate([ph_summary(standardized_cloud(a)) for a in mm.hidden_acts(P)]),
        "inv_poly_frobenius": lambda mm: np.array([np.sum(W ** 2) for W in mm.Ws]),
        "inv_poly_gram_det": lambda mm: np.array([np.linalg.det(W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W) for W in mm.Ws]),
        "inv_poly_roundtrip": lambda mm: np.array([np.sum(weight_product(mm) ** 2)]),
        "loop_ratios": loop_ratios,
        "singular_values": lambda mm: np.concatenate([np.linalg.svd(W, compute_uv=False) for W in mm.Ws]),
        "per_unit_activation": lambda mm: np.concatenate([a.mean(0) for a in mm.hidden_acts(P)]),
        "per_unit_gradxinput": lambda mm: _gradxinput(mm, P),
        "topk_units": lambda mm: np.sort(np.argsort(-np.abs(_gradxinput(mm, P)))[:3]).astype(float),   # sorted unit indices
        "weight_norms": lambda mm: np.concatenate([np.sqrt((W ** 2).sum(0) + b ** 2) for W, b in zip(mm.Ws[:-1], mm.bs[:-1])]),
    }
    base_clouds = m.hidden_acts(P)
    def unit_act_hung(mm, l):        # blind: match units to the base model by their activation columns
        return _hungarian_cols(mm.hidden_acts(P)[l], base_clouds[l]).mean(0)
    def unit_act_hung_std(mm, l):    # blind matching + per-unit standardisation (the Z5 repair)
        return standardized_cloud(_hungarian_cols(mm.hidden_acts(P)[l], base_clouds[l])).mean(0)
    def top_dir_proc(mm, l):         # blind: orthogonal Procrustes of the cloud onto the base cloud, then PCA direction
        A = _procrustes_cols(mm.hidden_acts(P)[l], base_clouds[l]); A = A - A.mean(0)
        v = np.linalg.svd(A, full_matrices=False)[2][0]
        return v * np.sign(v[np.argmax(np.abs(v))])
    def top_dir_hung_std(mm, l):     # blind matching + standardisation, then PCA direction
        A = standardized_cloud(_hungarian_cols(mm.hidden_acts(P)[l], base_clouds[l])); A = A - A.mean(0)
        v = np.linalg.svd(A, full_matrices=False)[2][0]
        return v * np.sign(v[np.argmax(np.abs(v))])
    for l in range(d):
        M[f"top_direction_L{l}"] = (lambda mm, l=l: top_direction(mm, l, 1)[:, 0])
        M[f"top_direction_L{l}_procrustes"] = (lambda mm, l=l: top_dir_proc(mm, l))
        M[f"top_direction_L{l}_hungarian_std"] = (lambda mm, l=l: top_dir_hung_std(mm, l))
        M[f"per_unit_activation_L{l}"] = (lambda mm, l=l: mm.hidden_acts(P)[l].mean(0))            # + exact gauge alignment
        M[f"per_unit_activation_L{l}_hungarian"] = (lambda mm, l=l: unit_act_hung(mm, l))
        M[f"per_unit_activation_L{l}_hungarian_std"] = (lambda mm, l=l: unit_act_hung_std(mm, l))
    return M


def job(args):
    spec, path, fn, do_f10 = args
    import torch; torch.set_num_threads(1)
    m = MLP.load(os.path.join(R, path)); X, y = truth_table(fn); P = _probe(X.shape[1])
    t0 = time.time()
    # F8: identity on the truth table for 100+100+100 renamings
    ident = {}
    for kind in ("perm", "scale", "combined"):
        rng = np.random.default_rng(hash((spec, kind)) % (1 << 31)); worst = 0.0
        for r in range(N_REP):
            g = S.random_element(m, rng, kind=kind, log_range=(0.1, 10.0))
            worst = max(worst, S.output_deviation(m, g, X))
        ident[kind] = worst
    f8 = dict(specimen=spec, max_dev_perm=ident["perm"], max_dev_scale=ident["scale"], max_dev_combined=ident["combined"],
              identity_ok=int(max(ident.values()) < 1e-6))
    # F9: anchor
    a = check_invariance(weight_product, m, n=N_REP, name="weight_product", uniform_sweep=False, keep_samples=False)
    f9 = dict(specimen=spec, cv_perm=a.median_cv["perm"], cv_scale=a.median_cv["scale"], cv_combined=a.median_cv["combined"],
              anchor_ok=int(max(a.median_cv.values()) < 1e-6))
    rows = []
    if do_f10:
        for name, fn_ in metrics_for(m, X, P).items():
            kw = {}
            import re
            mo = re.fullmatch(r"per_unit_activation_L(\d+)", name)
            if mo:
                kw = dict(align="gauge", unit_key=int(mo.group(1)), scale_power=1.0)   # exact alignment = ceiling
            try:
                res = check_invariance(lambda mm, f=fn_: np.atleast_1d(np.asarray(f(mm), dtype=np.float64)),
                                       m, n=N_REP, name=name, keep_samples=False, **kw)
                row = res.row(); row["specimen"] = spec; row["error"] = ""
            except Exception as e:
                row = dict(metric=name, specimen=spec, error=repr(e)[:200])
            rows.append(row)
    return f8, f9, rows, time.time() - t0


def _append(path, rows, fields=None):
    if not rows: return
    fields = fields or sorted({k for r in rows for k in r}, key=lambda k: (k not in ("specimen", "metric"), k))
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})


F10_FIELDS = ["specimen", "metric", "verdict", "verdict_aligned", "cv_perm", "cv_scale", "cv_combined", "frac_inv_perm",
              "frac_inv_scale", "frac_inv_combined", "cv_perm_aligned", "cv_scale_aligned", "cv_combined_aligned",
              "scaling_exponent", "scaling_r2", "align", "n", "error"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--f10-seeds", type=int, default=5)
    a = ap.parse_args()
    spec = list(csv.DictReader(open(os.path.join(R, "specimens.csv"))))
    p8 = os.path.join(R, "F8_identity.csv")
    done = {r["specimen"] for r in csv.DictReader(open(p8))} if os.path.exists(p8) else set()
    jobs = []
    for r in spec:
        if not int(r["converged"]): continue
        name = f'{r["function"]}_d{r["depth"]}_w{r["width"]}_s{r["seed"]}'
        if name in done: continue
        jobs.append((name, r["path"], r["function"], int(r["seed"]) < a.f10_seeds))
    jobs.sort(key=lambda j: not j[3])   # F10 specimens first
    print(f"{len(jobs)} specimens ({len(done)} done; F10 on {sum(j[3] for j in jobs)})", flush=True)
    with Pool(a.procs) as pool:
        for i, (f8, f9, rows, wall) in enumerate(pool.imap_unordered(job, jobs)):
            _append(p8, [f8]); _append(os.path.join(R, "F9_anchor.csv"), [f9]); _append(os.path.join(R, "audit_table.csv"), rows, F10_FIELDS)
            if i % 25 == 0: print(f"{i}/{len(jobs)} {f8['specimen']} wall {wall:.1f}s F8 {f8['identity_ok']} F9 {f9['anchor_ok']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
