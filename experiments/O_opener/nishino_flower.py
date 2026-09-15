#!/usr/bin/env python3
"""O1.3 Nishino Flower — T1 planted-bias detection (P3), SPEC_O.md §O1.3 / FINAL_SPEC §6.4.

  python nishino_flower.py [--specimens grok_planted_s0 grok_clean_s0] [--every 5] [--zz-null 10]

Per specimen it walks the checkpoint sequence and writes
  results/open/nishino_flower_<specimen>_clouds.npz   the gauge-fixed activation clouds + per-checkpoint logit statistics
                                                (the only GPU stage; cached so the detectors can be re-run without it)
  results/open/nishino_flower_<specimen>.json   trajectory + detectors + provenance (the evidence file; the cache
                                                that makes this script resumable — a specimen with a JSON is skipped)
and then rebuilds, from every JSON present,
  results/nishino_flower_trajectory.csv         one row per specimen x checkpoint
  results/nishino_flower_detections.csv         one row per specimen (ground truths, detector onsets, pass/fail, nulls)

Ground truths (both reported, never softened)
  * accuracy-level (SPEC_O): first checkpoint where ablating s (random residue token) moves test accuracy by > 5 points.
    May be undefined — AMENDMENTS #4 measured Δacc = 0.4 points at convergence on the recipe-1 planted specimen.
  * logit-level (AMENDMENTS #5): first checkpoint at f >= f_onset where the shortcut share σ = 1 − κ_f(leak) exceeds the
    matched-fraction clean reference R(f) by δ = 0.05 and stays above it.

Detectors (pre-registered; each is run in a `raw` variant — the literal SPEC_O pipeline, which must detect nothing on the
clean specimens — and a `diff` variant on leak-minus-marginalised, the shortcut-aligned refinement whose clean null is
vacuous by construction and is labelled as such):
  (a) zigzag persistence (whiteboxtopo.topology.zigzag) over the checkpoint sequence of MLP activation clouds at '=';
  (b) Laplacian eigenflow (whiteboxtopo.spectral.eigenflow) on the same clouds: gap and n_small;
  (c) a linear probe for the planted residue class (a+b) mod 8 on the same clouds, scored above the c-given Bayes
      baseline (knowing the answer c already pins (a+b) mod 8 to two candidates).

Gauge fixing (K-O-2): every cloud is per-unit standardised (each MLP unit divided by its own std over the probe set at
that checkpoint) before anything topological or spectral touches it — the exact repair E2/Z5 verified for per-unit ReLU
scalings — then projected on a fixed 20-dim basis and centred/scaled to unit median pairwise distance (Z AMENDMENTS #2).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys, time, traceback
import numpy as np
import torch
from scipy.spatial.distance import pdist

torch.set_num_threads(int(os.environ.get("NF_THREADS", 4)))     # the box is shared with 4 training units + an enumeration job

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import olib  # noqa: E402
from olib import RES  # noqa: E402
from sweep_tosho import P, N_SHORT, SHORT0, CK, fourier_kappa_f, marginal_logits  # noqa: E402
from whiteboxtopo import topology as T  # noqa: E402
from whiteboxtopo.topology.scaling import standardized_cloud  # noqa: E402
from whiteboxtopo.spectral import eigenflow  # noqa: E402
from whiteboxtopo.core.persistence import compute_diagrams  # noqa: E402

OUT = os.path.join(RES, "open"); os.makedirs(OUT, exist_ok=True)
TRAJ_CSV = os.path.join(RES, "nishino_flower_trajectory.csv")
DET_CSV = os.path.join(RES, "nishino_flower_detections.csv")

# ---- frozen constants (pre-registered; changes go to AMENDMENTS.md) ------------------------------------------------
N_PROBE = 256          # fixed probe pairs, the SAME ones at every checkpoint and in both channels (SPEC_O: 150-300)
N_PC = 20              # fixed linear basis: top PCs of the FINAL standardised leak cloud (Z1 convention)
N_SUB = 100            # zigzag subsample, lowered further under the simplex budget (Z AMENDMENTS #2/#3 ladder). O1.3
                       # starts at 100 rather than 150: at 150 points the O-specimen clouds sit at 85 k simplices — inside
                       # the budget — but Dionysus' zigzag took ~4 min on only 11 clouds, and the cost is superlinear
                       # (Z AMENDMENTS #3 measured the same and lowered N_SUB for wall time, not memory).
SIMPLEX_BUDGET = 30_000     # NOT the Z-block 800 k. Measured here (AMENDMENTS #6): at 91 370 simplices over 41 zigzag
                            # positions Dionysus had not finished after 12 min on this box, and at 84 983 over 21 positions
                            # it took > 4 min; the O specimens' beta1-maximising scale carries 45-78 H1 classes, and the
                            # zigzag cost grows with classes x positions, not with points. 30 k keeps a run to minutes.
ZZ_MAX_CK = 40         # zigzag cloud count cap -> onset resolution 1/40 = 2.5 % of training, inside the 5 % tolerance
ZZ_MIN_LIFE = 3        # a "long" H1 bar (Z1)
ZZ_MATCH = 2           # cloud indices within which a marginal-channel bar cancels a leak-channel one (diff variant)
KNN_K = 15; N_EIGS = 10                                  # spectral.eigenflow defaults (SPEC 2.10)
DELTA_LOGIT = 0.05     # AMENDMENTS #5 margin for the logit-level ground truth
DELTA_SENS = (0.02, 0.05, 0.10)
PROBE_EXCESS_MIN = 0.10   # shortcut-alignment guard on detector (c): the post-change probe accuracy must beat the
                          # c-given Bayes baseline by >= 10 points, else the change point is grokking, not a shortcut
ALPHA = 0.05           # permutation-null significance for every change point
N_PERM = 2000          # permutations for the 1-D change-point nulls
TOL = 0.05             # SPEC_O pass tolerance: |detected - ground truth| <= 5 % of training
# recipe-1 planted specimen that never adopted its shortcut (AMENDMENTS #4): an explicit null, like the clean specimens
INERT_CONTROLS = {"grok_planted30_s0_v1"}

DETECTORS = ["zigzag", "eigen_gap", "probe_res8", "probe_res8_v2", "probe_res8_v3"]
VARIANTS = ["raw", "diff"]
DET_VARIANTS = {"zigzag": ["raw", "diff"], "eigen_gap": ["raw", "diff"], "probe_res8": ["raw", "diff"],
                "probe_res8_v2": ["diff"],          # v2 IS the leak-marginal differential (AMENDMENTS #9)
                "probe_res8_v3": ["diff"]}          # v3 = the v2 rule on f > 0 only (AMENDMENTS #10)
N_PERM_LABEL = 2000    # label-permutation draws for the v2 null (AMENDMENTS #9)
RIDGE_LAMBDA = 1.0     # v2 closed-form probe


# ====================================================================================== clouds, logits, probe targets
@torch.no_grad()
def mlp_cloud(m, toks, batch=4096):
    """MLP hidden activations at the '=' position, (n, d_mlp) — the same nodes olib.NodeView exposes as `mlp*`."""
    out = []
    for i in range(0, len(toks), batch):
        _, cache = m(toks[i:i + batch], return_cache=True)
        out.append(cache["mlp"][-1][:, -1].float().cpu())
    return torch.cat(out).numpy()


def probe_tokens(data, idx, s_value=None, device=olib.DEV):
    """Tokens for pairs `idx`; planted: s = leaked residue token (s_value None) or the constant shortcut token s_value."""
    a, b = data["a"][idx], data["b"][idx]; cols = [a, b]
    if data["planted"]:
        cols.append(data["s_leak"][idx] if s_value is None else np.full_like(a, s_value))
    cols.append(np.full_like(a, P))
    return torch.as_tensor(np.stack(cols, 1), dtype=torch.long, device=device)


def clouds_at(m, data, idx):
    """(leak cloud, marginalised cloud) at the '=' position. The marginalised cloud is the activation averaged over the
    8 shortcut values — the activation-space analogue of AMENDMENTS #3's marginalised logits. Identical to the leak
    cloud for clean/memorising specimens (no shortcut token), which is recorded, not hidden."""
    A = mlp_cloud(m, probe_tokens(data, idx))
    if not data["planted"]:
        return A, A.copy()
    M = np.mean([mlp_cloud(m, probe_tokens(data, idx, s_value=s0)) for s0 in range(SHORT0, SHORT0 + N_SHORT)], 0)
    return A, M


def gauge_fix(A):
    """K-O-2 gauge fixing: divide every MLP unit by its own std over the probe set (exactly invariant under the per-unit
    ReLU scalings of G; topology.scaling.standardized_cloud)."""
    return standardized_cloud(A, how="std")


def z_normalise(c):
    """Centre and scale to unit median pairwise distance (Z AMENDMENTS #2): shape across training, not size."""
    c = c - c.mean(0); return c / max(float(np.median(pdist(c))), 1e-12)


def c_baseline(data, idx):
    """Accuracy of the Bayes-optimal predictor of r8 = (a+b) mod 8 from the answer c alone, fitted on the full p^2 grid
    and evaluated on the probe set. c leaves exactly two candidates for r8, so this is the level a model that merely
    solved the task already reaches — the probe detector is scored above it."""
    a, b = data["a"], data["b"]; c = (a + b) % P; r8 = (a + b) % N_SHORT
    tab = np.zeros((P, N_SHORT), int)
    np.add.at(tab, (c, r8), 1)
    rule = tab.argmax(1)
    cp, rp = (data["a"][idx] + data["b"][idx]) % P, (data["a"][idx] + data["b"][idx]) % N_SHORT
    return float((rule[cp] == rp).mean()), rule


def probe_accuracy(X, y, seed=0, folds=4):
    """Held-out accuracy of a multinomial logistic probe for y on the cloud X (stratified k-fold, fixed seed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(max_iter=300, C=1.0)      # lbfgs multinomial (sklearn >= 1.7 has no multi_class arg)
        clf.fit(X[tr], y[tr]); accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def fixed_folds(y, folds=4, seed=0):
    """The fold assignment v1 uses, frozen once from the true labels so that a label permutation changes only the
    labels (AMENDMENTS #9)."""
    from sklearn.model_selection import StratifiedKFold
    return [(tr, te) for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)]


def ridge_probe_setup(X, splits, lam=RIDGE_LAMBDA):
    """Per-fold hat matrices H = X_te (X_tr'X_tr + lam I)^-1 X_tr' (labels not involved), so a label permutation costs
    one matrix product per fold. Intercept via an appended ones column."""
    X = np.asarray(X, float); A = np.hstack([X, np.ones((len(X), 1))])
    out = []
    for tr, te in splits:
        Xtr, Xte = A[tr], A[te]
        G = Xtr.T @ Xtr + lam * np.eye(A.shape[1])
        out.append((te, Xte @ np.linalg.solve(G, Xtr.T), tr))
    return out


def ridge_probe_acc(setup, y, n_class=N_SHORT):
    """4-fold held-out accuracy of the closed-form ridge probe for labels y on a cloud prepared by ridge_probe_setup."""
    Y = np.eye(n_class)[np.asarray(y, int)]
    correct = 0; n = 0
    for te, H, tr in setup:
        pred = (H @ Y[tr]).argmax(1)
        correct += int((pred == y[te]).sum()); n += len(te)
    return correct / max(n, 1)


# ================================================================================================= change-point rules
def max_t(x):
    """Two-sample max-t change point: stat = max over splits i of |mean(x[i:]) - mean(x[:i])| / (s_p sqrt(1/i+1/(T-i))),
    index = the argmax split (first checkpoint of the post-change segment)."""
    x = np.asarray(x, float); Tn = len(x)
    if Tn < 6 or float(np.nanstd(x)) <= 0 or not np.isfinite(x).all():
        return 0.0, -1
    i = np.arange(1, Tn); cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    n1 = i.astype(float); n2 = (Tn - i).astype(float)
    m1 = cs[i - 1] / n1; m2 = (cs[-1] - cs[i - 1]) / n2
    v1 = np.maximum(cs2[i - 1] / n1 - m1 ** 2, 0.0); v2 = np.maximum((cs2[-1] - cs2[i - 1]) / n2 - m2 ** 2, 0.0)
    sp = np.sqrt(np.maximum((n1 * v1 + n2 * v2) / max(Tn - 2, 1), 1e-300))
    t = np.abs(m2 - m1) / (sp * np.sqrt(1 / n1 + 1 / n2))
    j = int(np.nanargmax(t)); return float(t[j]), int(i[j])


def changepoint(x, frac, n_perm=N_PERM, seed=0):
    """max-t change point + permutation p-value (shuffle checkpoint order: the null is 'the trace has no time
    structure'), and the pre-stated onset estimate: the first checkpoint at which the trace crosses the midpoint between
    the pre-change and post-change means (the half-rise convention of spectral.flow.crystallization_index, SPEC §6.8)."""
    x = np.asarray(x, float); out = dict(stat=0.0, p=1.0, cp_index=-1, onset_index=-1, onset_frac=None,
                                         pre=None, post=None, n=int(len(x)))
    stat, i = max_t(x)
    out["stat"] = stat; out["cp_index"] = i
    if i < 0:
        return out
    rng = np.random.default_rng(seed)
    null = np.array([max_t(rng.permutation(x))[0] for _ in range(n_perm)])
    out["p"] = float((1 + np.sum(null >= stat)) / (n_perm + 1))
    pre, post = float(x[:i].mean()), float(x[i:].mean()); out["pre"], out["post"] = pre, post
    mid = 0.5 * (pre + post); sgn = 1.0 if post >= pre else -1.0
    hit = np.flatnonzero(sgn * (x[:i + 1] - mid) >= 0)
    j = int(hit[0]) if hit.size else i
    out["onset_index"] = j; out["onset_frac"] = float(frac[j])
    return out


# ============================================================================================================ zigzag
def informative_scale(X):
    """Rips scale at which the final normalised cloud carries the most H1 classes (quantile 0.1 if it has none) —
    Z AMENDMENTS #2(b). Returns (scale, distance quantile, max beta1)."""
    D = pdist(X); dg = np.asarray(compute_diagrams(X, maxdim=1, backend="ripser", thresh=1.0).dgms[1]).reshape(-1, 2)
    if len(dg) == 0:
        return float(np.quantile(D, 0.1)), 0.1, 0
    grid = np.linspace(float(D.min()), 1.0, 200)
    b1 = np.array([np.sum((dg[:, 0] <= s) & (dg[:, 1] > s)) for s in grid])
    s = float(grid[int(np.argmax(b1))]); return s, float(np.mean(D <= s)), int(b1.max())


def estimate_simplices(sub, scale):
    """Cheap upper-bound screen before the authoritative dionysus count: edge density of the largest union times the
    triangle count of a complete complex. Only used to skip an n_sub that is obviously over budget."""
    worst = 0.0
    for i in range(len(sub) - 1):
        X = np.vstack([sub[i], sub[i + 1]]); n = len(X)
        dens = float((pdist(X) <= scale).mean())
        worst = max(worst, n + dens * n * (n - 1) / 2 + (dens ** 3) * n * (n - 1) * (n - 2) / 6)
    return worst * (len(sub) - 1)


def count_simplices(sub, scale, cap):
    import dionysus as d
    if estimate_simplices(sub, scale) > 4 * cap:            # obviously over: do not pay for fill_rips
        return int(estimate_simplices(sub, scale)), True
    tot = 0
    for i in range(len(sub) - 1):
        tot += len(d.fill_rips(np.vstack([sub[i], sub[i + 1]]), 2, float(scale)))
        if tot > cap:
            return tot, True
    return tot, False


ZZ_MIN_CK = 20         # floor for the cloud-count fallback; below this the onset resolution (1/19 = 5.3 %) would exceed
                       # the SPEC_O pass tolerance, so the run is reported as resolution-limited rather than coarsened further


def zigzag_summary(clouds, fracs, seed=0):
    """Zigzag over the checkpoint clouds at the informative scale, under the simplex budget (AMENDMENTS #6). The budget
    is met by lowering the point subsample first (Z AMENDMENTS #2/#3 ladder) and, only if that ladder is exhausted, by
    halving the number of clouds down to ZZ_MIN_CK. Returns the qualifying H1 classes (born after the first cloud, alive
    at the last, lifetime >= ZZ_MIN_LIFE positions), the traces, and `ck_sel` = which of the passed clouds were used."""
    rng = np.random.default_rng(seed); pick = rng.choice(len(clouds[0]), min(N_SUB, len(clouds[0])), replace=False)
    fracs = np.asarray(fracs, float)
    levels = [len(clouds)]
    while levels[-1] > ZZ_MIN_CK:
        levels.append(max(ZZ_MIN_CK, levels[-1] // 2))
    over = True
    for n_ck in levels:
        ck_sel = np.unique(np.linspace(0, len(clouds) - 1, n_ck).round().astype(int))
        cs = [clouds[i] for i in ck_sel]
        for n_sub in (N_SUB, 75, 50, 40, 30):
            sub = [c[pick[:n_sub]] for c in cs]
            scale, scale_q, b1max = informative_scale(sub[-1])
            t0 = time.time(); n_simp, over = count_simplices(sub, scale, SIMPLEX_BUDGET)
            print(f"    zigzag budget: n_ck {len(ck_sel)} n_sub {n_sub} scale {scale:.3f} q {scale_q:.3f} "
                  f"beta1max {b1max} simplices {n_simp} over={over} [{time.time() - t0:.0f}s]", flush=True)
            if not over:
                break
        if not over:
            break
    clouds = cs; fracs = fracs[ck_sel]
    t0 = time.time()
    z = T.zigzag(sub, scale, maxdim=1)
    print(f"    zigzag done: {len(clouds)} clouds, {z.n_simplices} simplices [{time.time() - t0:.0f}s]", flush=True)
    bars = np.asarray(z.bars[1]).reshape(-1, 2)
    life = (np.where(np.isfinite(bars[:, 1]), bars[:, 1], z.n_positions) - bars[:, 0]) if len(bars) else np.zeros(0)
    ci = z.bars_in_cloud_index(1) if len(bars) else np.zeros((0, 2))
    last = len(clouds) - 1
    qual = []
    for k in range(len(bars)):
        b_ck, d_ck = int(ci[k, 0]), int(ci[k, 1])
        if b_ck >= 1 and d_ck >= last and life[k] >= ZZ_MIN_LIFE:
            qual.append(dict(birth_ck=b_ck, birth_frac=float(fracs[b_ck]), life=float(life[k])))
    qual.sort(key=lambda r: -r["life"])
    births = np.zeros(len(clouds), int)
    for q in qual:
        births[q["birth_ck"]] += 1
    return dict(scale=float(scale), scale_quantile=float(scale_q), beta1_final_max=int(b1max), n_sub=int(n_sub),
                budget_reduced=int(n_sub < N_SUB), over_budget=int(over), n_clouds=int(len(clouds)),
                ck_sel=ck_sel.tolist(), clouds_reduced=int(len(ck_sel) < levels[0]),
                resolution_frac=float(1.0 / max(len(ck_sel) - 1, 1)), n_simplices=int(z.n_simplices), n_bars_H1=int(len(bars)),
                n_bars_H0=int(len(np.asarray(z.bars[0]).reshape(-1, 2))),
                max_life_not_initial=float(max([life[k] for k in range(len(bars)) if ci[k, 0] >= 1], default=0.0)),
                qualifying=qual, births_per_ck=births.tolist(),
                betti1_curve=np.asarray(z.betti_curve(1))[::2].tolist(),        # even positions = the clouds themselves
                betti0_curve=np.asarray(z.betti_curve(0))[::2].tolist())


def zigzag_null(clouds, scale, n_sub, n_perm, seed=0):
    """Permutation null for the zigzag statistic: shuffle the checkpoint ORDER (same clouds, no time structure) and
    recompute the longest non-initially-born H1 bar. Returns (p, null values). n_perm = 0 -> not run."""
    if n_perm <= 0:
        return None, []
    rng = np.random.default_rng(seed); pick = rng.choice(len(clouds[0]), min(n_sub, len(clouds[0])), replace=False)
    sub = [c[pick] for c in clouds]; vals = []
    for b in range(n_perm):
        order = np.random.default_rng(1000 + b).permutation(len(sub))
        z = T.zigzag([sub[i] for i in order], scale, maxdim=1)
        bars = np.asarray(z.bars[1]).reshape(-1, 2)
        if not len(bars):
            vals.append(0.0); continue
        life = np.where(np.isfinite(bars[:, 1]), bars[:, 1], z.n_positions) - bars[:, 0]
        ci = z.bars_in_cloud_index(1)
        vals.append(float(max([life[k] for k in range(len(bars)) if ci[k, 0] >= 1], default=0.0)))
    return None, [float(v) for v in vals]


# ======================================================================================= per-specimen trajectory pass
def forward_pass(name, every=1, verbose=True):
    """Stage 1 (the only GPU work): per-checkpoint logit statistics and the two gauge-fixed activation clouds.
    Cached in results/open/nishino_flower_<name>_clouds.npz so the detector stage can be re-run without the GPU."""
    t0 = time.time()
    m0, ck0, data = olib.load_grok(name)
    cfg = ck0["cfg"]; planted = bool(data["planted"])
    cks = olib.checkpoints(name)
    if every > 1:
        cks = cks[::every] + ([cks[-1]] if cks[-1] not in cks[::every] else [])
    total = float(cks[-1][0]) or 1.0
    onset_ep = cfg.get("onset"); f_onset = (float(onset_ep) / total) if onset_ep else None
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(data["te"], N_PROBE, replace=False))      # the SAME probe pairs at every checkpoint
    r8 = ((data["a"][idx] + data["b"][idx]) % N_SHORT).astype(int)
    base_c, _ = c_baseline(data, idx)
    te = data["te"]

    rows, raw_leak, raw_marg = [], [], []
    for step, _ in cks:
        m, ck, _ = olib.load_grok(name, step)
        L = olib.grok_logits(m, data)                                   # all p^2 pairs, leaked s for planted
        kf_leak = fourier_kappa_f(L)[0]
        acc_leak = float((L.argmax(1)[te] == data["c"][te]).mean())
        if planted:
            Lm = marginal_logits(m, data["a"], data["b"])
            kf_marg = fourier_kappa_f(Lm)[0]
            acc_marg = float((Lm.argmax(1)[te] == data["c"][te]).mean())
            acc_abl = float((olib.grok_logits(m, data, shortcut="random").argmax(1)[te] == data["c"][te]).mean())
        else:
            kf_marg, acc_marg, acc_abl = kf_leak, acc_leak, acc_leak
        A, M = clouds_at(m, data, idx)
        raw_leak.append(gauge_fix(A)); raw_marg.append(gauge_fix(M))
        rows.append(dict(epoch=int(step), frac=float(step) / total, acc_leak=acc_leak, acc_marginal=acc_marg,
                         acc_ablate=acc_abl, d_acc_ablate_pts=100.0 * (acc_leak - acc_abl),
                         kappa_f_leak=float(kf_leak), kappa_f_marginal=float(kf_marg),
                         shortcut_share=float(1.0 - kf_leak)))
        del m
        if verbose and len(rows) % 20 == 0:
            print(f"  {name} ckpt {len(rows)}/{len(cks)} ep {step} acc {acc_leak:.3f} kf {kf_leak:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    torch.cuda.empty_cache()
    print(f"  {name}: forward pass done ({len(rows)} ckpt) [{time.time() - t0:.0f}s]", flush=True)

    # fixed 20-dim basis from the FINAL standardised leak cloud, then Z-normalise every cloud (Z AMENDMENTS #2)
    Xf = raw_leak[-1]; mu = Xf.mean(0)
    _, _, Vt = np.linalg.svd(Xf - mu, full_matrices=False); B = Vt[:N_PC].T
    cl_leak = np.stack([z_normalise((c - mu) @ B) for c in raw_leak])
    cl_marg = np.stack([z_normalise((c - mu) @ B) for c in raw_marg])
    meta = dict(specimen=name, condition=cfg["condition"], seed=cfg["seed"], recipe=cfg.get("recipe", 1),
                planted=planted, total_epochs=total, onset_epoch=onset_ep, onset_frac=f_onset, rows=rows,
                c_baseline=base_c, r8=r8.tolist(), n_probe=N_PROBE, every=every, wall_forward=round(time.time() - t0, 1))
    np.savez_compressed(cloud_cache(name), cl_leak=cl_leak, cl_marg=cl_marg, meta=json.dumps(meta))
    return meta, cl_leak, cl_marg


def cloud_cache(name):
    return os.path.join(OUT, f"nishino_flower_{name}_clouds.npz")


def trajectory(name, every=1, zz_null=0, verbose=True, reuse=True):
    """Stage 2: the detectors, on the cached (or freshly computed) gauge-fixed clouds. Writes the evidence JSON."""
    t0 = time.time()
    cache = cloud_cache(name)
    if reuse and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"])); cl_leak = z["cl_leak"]; cl_marg = z["cl_marg"]
        print(f"  {name}: reusing cached clouds ({len(meta['rows'])} ckpt, {cache})", flush=True)
    else:
        meta, cl_leak, cl_marg = forward_pass(name, every=every, verbose=verbose)
    rows = meta["rows"]; planted = bool(meta["planted"]); name = meta["specimen"]
    base_c = float(meta["c_baseline"]); r8 = np.asarray(meta["r8"], dtype=int)
    total = float(meta["total_epochs"]); onset_ep = meta["onset_epoch"]; f_onset = meta["onset_frac"]
    cfg = dict(condition=meta["condition"], seed=meta["seed"], recipe=meta.get("recipe", 1))
    cl_leak = list(cl_leak); cl_marg = list(cl_marg)
    frac = np.array([r["frac"] for r in rows])

    # ---- (c) linear probe for the planted residue class -------------------------------------------------------
    for i, r in enumerate(rows):
        r["probe_acc_leak"] = probe_accuracy(cl_leak[i], r8)
        r["probe_acc_marginal"] = probe_accuracy(cl_marg[i], r8) if planted else r["probe_acc_leak"]
        r["probe_excess_leak"] = r["probe_acc_leak"] - base_c
        r["probe_excess_marginal"] = r["probe_acc_marginal"] - base_c
    print(f"  {name}: probes done [{time.time() - t0:.0f}s]", flush=True)
    perm_r8 = np.random.default_rng(7).permutation(r8)
    probe_label_null = probe_accuracy(cl_leak[-1], perm_r8)             # label-permutation null at the final cloud

    # ---- (c-v2) the leak - marginal probe differential + its 2000-draw label-permutation null (AMENDMENTS #9) ---
    splits = fixed_folds(r8, folds=4, seed=0)
    set_l = [ridge_probe_setup(c, splits) for c in cl_leak]
    set_m = [ridge_probe_setup(c, splits) for c in cl_marg] if planted else set_l
    rl = np.array([ridge_probe_acc(s, r8) for s in set_l]); rm = np.array([ridge_probe_acc(s, r8) for s in set_m])
    d_obs = rl - rm
    for i, r in enumerate(rows):
        r["probe_v2_acc_leak"] = float(rl[i]); r["probe_v2_acc_marginal"] = float(rm[i]); r["probe_v2_diff"] = float(d_obs[i])
    r_ridge_log_leak = float(np.corrcoef(rl, [r["probe_acc_leak"] for r in rows])[0, 1]) if np.std(rl) > 0 else float("nan")
    r_ridge_log_marg = float(np.corrcoef(rm, [r["probe_acc_marginal"] for r in rows])[0, 1]) if np.std(rm) > 0 else float("nan")
    pos = np.array([r["epoch"] > 0 for r in rows])            # AMENDMENTS #10: the f > 0 sub-trace for v3
    stat_obs, _ = max_t(d_obs)
    stat_obs3, _ = max_t(d_obs[pos])
    null_stats, null_stats3 = [], []
    if planted:
        rng_lab = np.random.default_rng(11)
        for b in range(N_PERM_LABEL):
            yp = rng_lab.permutation(r8)
            dn = np.array([ridge_probe_acc(s, yp) for s in set_l]) - np.array([ridge_probe_acc(s, yp) for s in set_m])
            null_stats.append(max_t(dn)[0]); null_stats3.append(max_t(dn[pos])[0])   # same draws, matched slicing
        print(f"  {name}: probe v2/v3 label-permutation null done ({N_PERM_LABEL} draws) [{time.time() - t0:.0f}s]", flush=True)
    p_v2 = (float((1 + sum(v >= stat_obs for v in null_stats)) / (len(null_stats) + 1)) if null_stats else 1.0)
    p_v3 = (float((1 + sum(v >= stat_obs3 for v in null_stats3)) / (len(null_stats3) + 1)) if null_stats3 else 1.0)

    # ---- (b) Laplacian eigenflow ------------------------------------------------------------------------------
    ef_l = eigenflow(cl_leak, k=KNN_K, n_eigs=N_EIGS)
    ef_m = eigenflow(cl_marg, k=KNN_K, n_eigs=N_EIGS) if planted else ef_l
    for i, r in enumerate(rows):
        r["eig_gap_leak"] = float(ef_l.gap[i]); r["eig_n_small_leak"] = int(ef_l.n_small[i])
        r["eig_gap_marginal"] = float(ef_m.gap[i]); r["eig_n_small_marginal"] = int(ef_m.n_small[i])

    print(f"  {name}: eigenflow done [{time.time() - t0:.0f}s]", flush=True)

    # ---- (a) zigzag -------------------------------------------------------------------------------------------
    sel = np.unique(np.linspace(0, len(rows) - 1, min(ZZ_MAX_CK, len(rows))).round().astype(int))
    zf = frac[sel]
    zz_l = zigzag_summary([cl_leak[i] for i in sel], zf, seed=0)
    zz_m = zigzag_summary([cl_marg[i] for i in sel], zf, seed=0) if planted else dict(zz_l, qualifying=list(zz_l["qualifying"]))
    for r in rows:
        r["zz_betti1_leak"] = ""; r["zz_betti1_marginal"] = ""; r["zz_births_leak"] = ""; r["zz_births_marginal"] = ""
    for j, k in enumerate(zz_l["ck_sel"]):
        i = int(sel[k]); rows[i]["zz_betti1_leak"] = int(zz_l["betti1_curve"][j]); rows[i]["zz_births_leak"] = int(zz_l["births_per_ck"][j])
    for j, k in enumerate(zz_m["ck_sel"]):
        i = int(sel[k]); rows[i]["zz_betti1_marginal"] = int(zz_m["betti1_curve"][j]); rows[i]["zz_births_marginal"] = int(zz_m["births_per_ck"][j])
    if zz_null > 0:
        _, nullvals = zigzag_null([cl_leak[int(sel[k])] for k in zz_l["ck_sel"]], zz_l["scale"], zz_l["n_sub"], zz_null)
        obs = zz_l["max_life_not_initial"]
        zz_l["null_max_life"] = nullvals
        zz_l["p"] = float((1 + sum(v >= obs for v in nullvals)) / (len(nullvals) + 1))
    else:
        zz_l["null_max_life"] = []; zz_l["p"] = None
        zz_l["p_note"] = ("no permutation null run (--zz-null 0). Each permutation is one full zigzag (1-2 min per channel "
                          "here), and alpha = 0.05 needs at least 19, i.e. 20-40 min per specimen per channel on this "
                          "shared box. The zigzag detection rule is structural (a surviving long H1 class), not a "
                          "threshold on a p-value, so the verdict does not depend on it; the null is still owed.")

    # ---- detector verdicts ------------------------------------------------------------------------------------
    det = {}
    # (a) zigzag: onset = birth of the LONGEST qualifying H1 class (born after cloud 0, alive at the end, life >= 3)
    q = zz_l["qualifying"]
    zz_raw = dict(detected=int(bool(q)), onset_frac=(q[0]["birth_frac"] if q else None),
                  stat=(q[0]["life"] if q else 0.0), p=zz_l["p"],
                  rule=f"longest H1 class born at cloud index >= 1, alive at the last cloud, lifetime >= {ZZ_MIN_LIFE} positions",
                  n_qualifying=len(q), earliest_frac=(min(r["birth_frac"] for r in q) if q else None))
    qm = zz_m["qualifying"]
    cancelled = bool(q) and any(abs(r["birth_ck"] - q[0]["birth_ck"]) <= ZZ_MATCH for r in qm)
    zz_diff = dict(detected=int(bool(q) and not cancelled), onset_frac=(q[0]["birth_frac"] if q and not cancelled else None),
                   stat=(q[0]["life"] if q else 0.0), p=zz_l["p"],
                   rule=f"the raw rule in the leak channel with no qualifying marginal-channel class born within {ZZ_MATCH} cloud indices",
                   cancelled_by_marginal=int(cancelled), vacuous_null=int(not planted))
    det["zigzag"] = dict(raw=zz_raw, diff=zz_diff)

    # (b) eigenflow gap: change point on the leak trace (raw) / on leak - marginal (diff)
    g_l = np.array([r["eig_gap_leak"] for r in rows]); g_m = np.array([r["eig_gap_marginal"] for r in rows])
    cp = changepoint(g_l, frac); cp["detected"] = int(cp["p"] < ALPHA and cp["onset_frac"] is not None)
    cp["rule"] = f"max-t change point on the spectral gap, permutation p < {ALPHA}; onset = half-rise crossing"
    cpd = changepoint(g_l - g_m, frac); cpd["detected"] = int(cpd["p"] < ALPHA and cpd["onset_frac"] is not None)
    cpd["rule"] = "the same rule on gap(leak) - gap(marginal)"; cpd["vacuous_null"] = int(not planted)
    det["eigen_gap"] = dict(raw=cp, diff=cpd)

    # (c) probe: change point on the excess over the c-given baseline, plus the shortcut-alignment guard
    e_l = np.array([r["probe_excess_leak"] for r in rows]); e_m = np.array([r["probe_excess_marginal"] for r in rows])
    cp = changepoint(e_l, frac)
    cp["post_excess"] = cp["post"]
    cp["detected"] = int(cp["p"] < ALPHA and cp["onset_frac"] is not None and (cp["post"] or -1) >= PROBE_EXCESS_MIN)
    cp["rule"] = (f"max-t change point on probe_acc(r8) - c-given Bayes baseline ({base_c:.3f}), permutation p < {ALPHA} "
                  f"AND post-change excess >= {PROBE_EXCESS_MIN}; onset = half-rise crossing")
    cpd = changepoint(e_l - e_m, frac)
    cpd["detected"] = int(cpd["p"] < ALPHA and cpd["onset_frac"] is not None and (cpd["post"] or -1) >= PROBE_EXCESS_MIN)
    cpd["rule"] = "the same rule on probe_acc(leak) - probe_acc(marginal)"; cpd["vacuous_null"] = int(not planted)
    det["probe_res8"] = dict(raw=cp, diff=cpd)

    # (c-v2) AMENDMENTS #9: change point on the differential, label-permutation p-value, no baseline guard
    cp2 = changepoint(d_obs, frac)                      # onset by the same half-rise rule as v1
    cp2["stat"] = stat_obs                              # the max-t statistic the label null is built for
    cp2["p_changepoint_order"] = cp2.pop("p")           # the order-permutation p is kept as evidence
    cp2["p"] = p_v2
    cp2["detected"] = int(p_v2 < ALPHA and cp2["onset_frac"] is not None and planted)
    cp2["rule"] = (f"max-t change point on probe_acc(leak) - probe_acc(marginal) (closed-form ridge probe, lambda "
                   f"{RIDGE_LAMBDA}, same fixed 4-fold split), p from a {N_PERM_LABEL}-draw label-permutation null of the "
                   f"whole differential trace, p < {ALPHA}; onset = half-rise crossing. No baseline guard.")
    cp2["vacuous_null"] = int(not planted)
    cp2["n_perm_label"] = len(null_stats)
    cp2["null_max"] = (float(np.max(null_stats)) if null_stats else None)
    cp2["null_q95"] = (float(np.quantile(null_stats, 0.95)) if null_stats else None)
    det["probe_res8_v2"] = dict(diff=cp2)

    # (c-v3) AMENDMENTS #10: the identical rule on the f > 0 sub-trace only
    cp3 = changepoint(d_obs[pos], frac[pos])
    cp3["stat"] = stat_obs3
    cp3["p_changepoint_order"] = cp3.pop("p")
    cp3["p"] = p_v3
    cp3["detected"] = int(p_v3 < ALPHA and cp3["onset_frac"] is not None and planted)
    cp3["rule"] = ("the probe v2 rule (AMENDMENTS #9) applied to checkpoints with f > 0 only; the untrained checkpoint is "
                   "excluded because the leaked token is read verbatim from the residual stream there (AMENDMENTS #10). "
                   f"{N_PERM_LABEL}-draw label-permutation null recomputed on the f > 0 trace, p < {ALPHA}.")
    cp3["vacuous_null"] = int(not planted)
    cp3["n_perm_label"] = len(null_stats3)
    cp3["n_ckpt_used"] = int(pos.sum())
    cp3["null_max"] = (float(np.max(null_stats3)) if null_stats3 else None)
    cp3["null_q95"] = (float(np.quantile(null_stats3, 0.95)) if null_stats3 else None)
    det["probe_res8_v3"] = dict(diff=cp3)

    # ---- ground truth 1: SPEC_O accuracy-level, as an entry/exit window (AMENDMENTS #7) ------------------------
    if planted:
        mask = np.array([(f_onset is None or r["frac"] >= f_onset) and abs(r["d_acc_ablate_pts"]) > 5.0 for r in rows])
        (gt_acc, gt_acc_ep), (gt_acc_x, gt_acc_x_ep) = sustained_window(mask, rows)
    else:
        gt_acc = gt_acc_ep = gt_acc_x = gt_acc_x_ep = None

    out = dict(specimen=name, condition=cfg["condition"], seed=cfg["seed"], recipe=cfg.get("recipe", 1), planted=planted,
               n_ckpt=len(rows), total_epochs=total, onset_epoch=onset_ep, onset_frac=f_onset,
               null_role=("inert_shortcut_null" if name in INERT_CONTROLS else ("clean_null" if not planted else "")),
               probe=dict(n_probe=N_PROBE, idx_seed=0, c_baseline=base_c, label_perm_null_final=probe_label_null,
                          target="(a+b) mod 8 = the planted residue class"),
               probe_v2=dict(estimator=f"closed-form ridge on one-hot r8, lambda={RIDGE_LAMBDA}, intercept, fixed 4-fold split",
                             ridge_vs_logistic_r=dict(leak=r_ridge_log_leak, marginal=r_ridge_log_marg),
                             n_perm_label=len(null_stats), stat=stat_obs,
                             acc_leak_final=float(rl[-1]), acc_marginal_final=float(rm[-1]),
                             diff_max=float(np.max(d_obs)), diff_argmax_frac=float(frac[int(np.argmax(d_obs))]),
                             diff_at_init=float(d_obs[0]), n_ckpt_pos=int(pos.sum()), stat_pos=stat_obs3,
                             diff_max_pos=float(np.max(d_obs[pos])),
                             diff_argmax_frac_pos=float(frac[pos][int(np.argmax(d_obs[pos]))])),
               gauge=dict(fixing="per-unit standardisation of the MLP cloud (topology.scaling.standardized_cloud, how='std'), "
                                 "per checkpoint, then a fixed %d-dim basis from the final standardised leak cloud, then "
                                 "centre + unit median pairwise distance (Z AMENDMENTS #2)" % N_PC,
                          justification="K-O-2: per-unit ReLU scalings are in G and E2 measured raw-cloud persistence to be "
                                        "non-invariant under them; standardisation is the exact repair, recorded as gauge-fixing "
                                        "(FINAL_SPEC §2.10 'per-unit scaling requires normalization (recorded as gauge-fixing)')",
                          exact=True),
               zigzag=dict(leak=zz_l, marginal=zz_m, n_clouds=int(zz_l["n_clouds"]), ck_indices=sel.tolist(),
                           resolution_frac=float(zz_l["resolution_frac"]), n_perm=int(zz_null)),
               eigenflow=dict(k=KNN_K, n_eigs=N_EIGS, threshold=float(ef_l.threshold)),
               detectors=det, gt_accuracy=dict(frac=gt_acc, epoch=gt_acc_ep, exit_frac=gt_acc_x, exit_epoch=gt_acc_x_ep,
                                               final_d_acc_pts=rows[-1]["d_acc_ablate_pts"],
                                               max_abs_d_acc_pts=float(max(abs(r["d_acc_ablate_pts"]) for r in rows)),
                                               rule="entry = first checkpoint at f >= data onset with |acc(leak) - acc(random s)| > 5 points, "
                                                    "sustained over 3 % of training; exit = the matching end of that window (SPEC_O + AMENDMENTS #7)",
                                               defined=gt_acc is not None),
               trajectory=rows, wall=round(time.time() - t0, 1),
               provenance=dict(script="experiments/O_opener/nishino_flower.py", checkpoints=len(rows), every=meta.get("every", every),
                               library=["whiteboxtopo.topology.zigzag", "whiteboxtopo.spectral.eigenflow",
                                        "whiteboxtopo.topology.scaling.standardized_cloud", "sklearn.LogisticRegression"],
                               cloud="MLP hidden activations at the '=' position (d_mlp = 512), probe set fixed across checkpoints",
                               marginal_channel=("mean activation over the 8 shortcut tokens (AMENDMENTS #3 analogue in "
                                                 "activation space)" if planted else "identical to the leak channel (no shortcut token)")))
    out["wall_forward"] = meta.get("wall_forward")
    json.dump(out, open(os.path.join(OUT, f"nishino_flower_{name}.json"), "w"), indent=1, default=str)
    return out


# ======================================================================================== aggregate / scoring pass
def clean_reference(runs):
    """R(f): median over included clean specimens of the shortcut share at matched training fraction (AMENDMENTS #5)."""
    inc = set()
    try:
        inc = {r["name"] for r in csv.DictReader(open(os.path.join(RES, "specimens.csv"))) if r["included"] == "1"}
    except Exception:
        pass
    cl = [r for r in runs if r["condition"] == "clean" and (not inc or r["specimen"] in inc)]
    if not cl:                                   # no included clean specimen yet: fall back, and say so in the CSV
        cl = [r for r in runs if r["condition"] == "clean"]
    if not cl:
        return None, []
    grid = np.linspace(0.0, 1.0, 201)
    curves = []
    for r in cl:
        f = np.array([t["frac"] for t in r["trajectory"]]); s = np.array([t["shortcut_share"] for t in r["trajectory"]])
        curves.append(np.interp(grid, f, s))
    return (grid, np.median(np.stack(curves), 0)), [r["specimen"] for r in cl]


PERSIST_FRAC = 0.03    # AMENDMENTS #7: a crossing is the entry once it holds for 3 % of training


def sustained_window(mask, rows):
    """AMENDMENTS #7. Entry = first index whose next W = max(2, ceil(0.03 n)) checkpoints all satisfy `mask`;
    exit = first index after the entry whose next W checkpoints all fail it (None = the condition never lapses).
    Returns ((entry_frac, entry_epoch), (exit_frac, exit_epoch)); a crossing starting inside the last W checkpoints
    cannot be certified and yields no entry."""
    mask = np.asarray(mask, bool); n = len(mask); W = max(2, int(np.ceil(PERSIST_FRAC * n)))
    entry = next((i for i in range(n - W + 1) if mask[i:i + W].all()), None)
    if entry is None:
        return (None, None), (None, None)
    ex = next((j for j in range(entry + 1, n - W + 1) if not mask[j:j + W].any()), None)
    f = lambda i: (float(rows[i]["frac"]), int(rows[i]["epoch"])) if i is not None else (None, None)   # noqa: E731
    return f(entry), f(ex)


def gt_logit_within(run, delta=DELTA_LOGIT):
    """AMENDMENTS #5 addendum: first checkpoint at f >= f_onset with sigma_within = kappa_f(marg) - kappa_f(leak) > delta,
    sustained. Architecture-matched (same model, same checkpoint), so it is silent on the inert-shortcut control."""
    if not run["planted"]:
        return (None, None), (None, None)
    rows = run["trajectory"]
    f = np.array([t["frac"] for t in rows])
    d = np.array([t["kappa_f_marginal"] - t["kappa_f_leak"] for t in rows])
    return sustained_window((d > delta) & (f >= (run["onset_frac"] or 0.0)), rows)


def gt_logit(run, ref, delta=DELTA_LOGIT):
    """First checkpoint at f >= f_onset with sigma(f) > R(f) + delta, holding at every later checkpoint (AMENDMENTS #5)."""
    if ref is None or not run["planted"]:
        return (None, None), (None, None)
    grid, R = ref; rows = run["trajectory"]
    f = np.array([t["frac"] for t in rows]); s = np.array([t["shortcut_share"] for t in rows])
    return sustained_window((s > np.interp(f, grid, R) + delta) & (f >= (run["onset_frac"] or 0.0)), rows)


def aggregate(verbose=True):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "nishino_flower_*.json")))]
    if not runs:
        print("no specimens processed"); return []
    ref, ref_names = clean_reference(runs)

    tf = ["specimen", "condition", "seed", "recipe", "null_role", "epoch", "frac", "acc_leak", "acc_marginal", "acc_ablate",
          "d_acc_ablate_pts", "kappa_f_leak", "kappa_f_marginal", "shortcut_share", "sigma_within", "clean_reference_share",
          "probe_acc_leak", "probe_acc_marginal", "probe_excess_leak", "probe_excess_marginal",
          "probe_v2_acc_leak", "probe_v2_acc_marginal", "probe_v2_diff",
          "eig_gap_leak", "eig_gap_marginal", "eig_n_small_leak", "eig_n_small_marginal",
          "zz_betti1_leak", "zz_betti1_marginal", "zz_births_leak", "zz_births_marginal"]
    with open(TRAJ_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tf); w.writeheader()
        for r in runs:
            for t in r["trajectory"]:
                row = dict(t); row.update(specimen=r["specimen"], condition=r["condition"], seed=r["seed"],
                                          recipe=r.get("recipe", 1), null_role=r.get("null_role", ""))
                row["sigma_within"] = t["kappa_f_marginal"] - t["kappa_f_leak"]
                row["clean_reference_share"] = (float(np.interp(t["frac"], ref[0], ref[1])) if ref else "")
                w.writerow({k: row.get(k, "") for k in tf})

    df = ["specimen", "condition", "seed", "recipe", "null_role", "n_ckpt", "n_ckpt_zigzag", "onset_frac_data",
          "gt_acc_frac", "gt_acc_epoch", "gt_acc_exit_frac", "gt_acc_exit_epoch", "gt_acc_defined",
          "gt_acc_final_delta_pts", "gt_acc_max_delta_pts",
          "gt_logit_frac", "gt_logit_epoch", "gt_logit_exit_frac", "gt_logit_exit_epoch", "gt_logit_defined",
          "gt_logit_delta", "gt_logit_sensitivity",
          "gt_logit_within_frac", "gt_logit_within_epoch", "gt_logit_within_exit_frac", "gt_logit_within_exit_epoch",
          "gt_logit_within_defined", "gt_logit_within_sensitivity",
          "clean_reference_from"]
    for d in DETECTORS:
        for v in DET_VARIANTS[d]:
            df += [f"{d}_{v}_detected", f"{d}_{v}_onset", f"{d}_{v}_onset_if_forced", f"{d}_{v}_stat", f"{d}_{v}_p",
                   f"{d}_{v}_absdelta_acc", f"{d}_{v}_pass_acc", f"{d}_{v}_absdelta_logit", f"{d}_{v}_pass_logit",
                   f"{d}_{v}_absdelta_logitw", f"{d}_{v}_pass_logitw",
                   f"{d}_{v}_false_detection", f"{d}_{v}_null_vacuous"]
    rows = []
    for r in runs:
        (gl, gle), (glx, glxe) = gt_logit(r, ref)
        (gw, gwe), (gwx, gwxe) = gt_logit_within(r)
        sens = {f"{d}": (gt_logit(r, ref, d)[0][0]) for d in DELTA_SENS}
        sensw = {f"{d}": (gt_logit_within(r, d)[0][0]) for d in DELTA_SENS}
        row = dict(specimen=r["specimen"], condition=r["condition"], seed=r["seed"], recipe=r.get("recipe", 1),
                   null_role=r.get("null_role", ""), n_ckpt=r["n_ckpt"], n_ckpt_zigzag=r["zigzag"]["n_clouds"],
                   onset_frac_data=r["onset_frac"], gt_acc_frac=r["gt_accuracy"]["frac"], gt_acc_epoch=r["gt_accuracy"]["epoch"],
                   gt_acc_exit_frac=r["gt_accuracy"].get("exit_frac"), gt_acc_exit_epoch=r["gt_accuracy"].get("exit_epoch"),
                   gt_acc_defined=int(r["gt_accuracy"]["defined"]), gt_acc_final_delta_pts=round(r["gt_accuracy"]["final_d_acc_pts"], 3),
                   gt_acc_max_delta_pts=round(r["gt_accuracy"]["max_abs_d_acc_pts"], 3),
                   gt_logit_frac=gl, gt_logit_epoch=gle, gt_logit_exit_frac=glx, gt_logit_exit_epoch=glxe,
                   gt_logit_defined=int(gl is not None), gt_logit_delta=DELTA_LOGIT,
                   gt_logit_sensitivity=json.dumps(sens), gt_logit_within_frac=gw, gt_logit_within_epoch=gwe,
                   gt_logit_within_exit_frac=gwx, gt_logit_within_exit_epoch=gwxe,
                   gt_logit_within_defined=int(gw is not None), gt_logit_within_sensitivity=json.dumps(sensw),
                   clean_reference_from=" ".join(ref_names))
        is_null = r.get("null_role", "") != ""
        for d in DETECTORS:
            for v in DET_VARIANTS[d]:
                c = r["detectors"].get(d, {}).get(v)
                if c is None:                       # a specimen processed before this detector existed
                    for suf in ("detected", "onset", "onset_if_forced", "stat", "p", "null_vacuous",
                                "absdelta_acc", "pass_acc", "absdelta_logit", "pass_logit",
                                "absdelta_logitw", "pass_logitw", "false_detection"):
                        row[f"{d}_{v}_{suf}"] = ""
                    continue
                on = c.get("onset_frac"); det = int(c.get("detected", 0))
                row[f"{d}_{v}_detected"] = det
                row[f"{d}_{v}_onset"] = (on if det else "")            # a non-firing detector reports no onset
                row[f"{d}_{v}_onset_if_forced"] = on                   # the change point it would have picked
                row[f"{d}_{v}_stat"] = (round(float(c["stat"]), 4) if c.get("stat") is not None else "")
                row[f"{d}_{v}_p"] = (round(float(c["p"]), 4) if c.get("p") is not None else "")
                row[f"{d}_{v}_null_vacuous"] = int(c.get("vacuous_null", 0))
                for gt, tag in ((r["gt_accuracy"]["frac"], "acc"), (gl, "logit"), (gw, "logitw")):
                    if det and on is not None and gt is not None:
                        ad = abs(on - gt); row[f"{d}_{v}_absdelta_{tag}"] = round(ad, 4)
                        row[f"{d}_{v}_pass_{tag}"] = int(ad <= TOL)
                    else:
                        row[f"{d}_{v}_absdelta_{tag}"] = ""; row[f"{d}_{v}_pass_{tag}"] = ("" if gt is None else 0)
                row[f"{d}_{v}_false_detection"] = (det if is_null else "")
        rows.append(row)
    with open(DET_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=df); w.writeheader(); w.writerows(rows)

    if verbose:
        print("\n=== O1.3 Nishino Flower — summary " + "=" * 50)
        print(f"clean reference from: {ref_names or 'NONE (logit-level ground truth undefined)'}")
        for row in rows:
            tagn = f" [{row['null_role']}]" if row["null_role"] else ""
            print(f"\n{row['specimen']}{tagn}  recipe {row['recipe']}  n_ckpt {row['n_ckpt']} (zigzag {row['n_ckpt_zigzag']})"
                  f"  data onset {row['onset_frac_data']}")
            print(f"  GT accuracy-level : entry {row['gt_acc_frac']} exit {row['gt_acc_exit_frac']}  "
                  f"(defined={row['gt_acc_defined']}; final Δacc {row['gt_acc_final_delta_pts']} pts, "
                  f"max |Δacc| {row['gt_acc_max_delta_pts']} pts)")
            print(f"  GT logit clean-ref: entry {row['gt_logit_frac']} exit {row['gt_logit_exit_frac']}  "
                  f"(defined={row['gt_logit_defined']}, δ={DELTA_LOGIT}; sensitivity {row['gt_logit_sensitivity']})")
            print(f"  GT logit within   : entry {row['gt_logit_within_frac']} exit {row['gt_logit_within_exit_frac']}  "
                  f"(defined={row['gt_logit_within_defined']}; sensitivity {row['gt_logit_within_sensitivity']})")
            for d in DETECTORS:
                for v in DET_VARIANTS[d]:
                    if row.get(f"{d}_{v}_detected", "") == "":
                        continue
                    vac = " VACUOUS-NULL" if row[f"{d}_{v}_null_vacuous"] else ""
                    fd = row[f"{d}_{v}_false_detection"]
                    print(f"  {d:10s} {v:4s}: detected={row[f'{d}_{v}_detected']} onset={row[f"{d}_{v}_onset"] if row[f"{d}_{v}_onset"] != "" else '-'} "
                          f"stat={row[f'{d}_{v}_stat']} p={row[f'{d}_{v}_p']} "
                          f"|Δ|acc={row[f'{d}_{v}_absdelta_acc']} pass_acc={row[f'{d}_{v}_pass_acc']} "
                          f"|Δ|logit={row[f'{d}_{v}_absdelta_logit']} pass_logit={row[f'{d}_{v}_pass_logit']} "
                          f"|Δ|logitW={row[f'{d}_{v}_absdelta_logitw']} pass_logitW={row[f'{d}_{v}_pass_logitw']}"
                          + (f" FALSE_DETECTION={fd}" if fd != "" else "") + vac)
        nulls = [r for r in rows if r["null_role"]]
        for d in DETECTORS:
            for v in DET_VARIANTS[d]:
                n_fd = sum(int(r.get(f"{d}_{v}_false_detection") or 0) for r in nulls)
                n_vac = sum(int(r.get(f"{d}_{v}_null_vacuous") or 0) for r in nulls)
                print(f"false detections on nulls ({len(nulls)} specimens, {n_vac} of them vacuous) "
                      f"{d:13s} {v:4s}: {n_fd}")
        print(f"\nwrote {TRAJ_CSV}\nwrote {DET_CSV}")
    return rows


def default_specimens():
    path = os.path.join(RES, "specimens.csv")
    rows = list(csv.DictReader(open(path)))
    return [r["name"] for r in rows if r["kind"] == "grok" and r["included"] == "1" and r["condition"] in ("clean", "planted")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimens", nargs="*", default=None)
    ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint (smoke runs)")
    ap.add_argument("--zz-null", type=int, default=0, help="zigzag cloud-order permutations (0 = skip; each costs one zigzag)")
    ap.add_argument("--n-sub", type=int, default=None, help="override the zigzag subsample size (default %d)" % N_SUB)
    ap.add_argument("--zz-max-ck", type=int, default=None, help="override the zigzag cloud cap (default %d)" % ZZ_MAX_CK)
    ap.add_argument("--force", action="store_true", help="recompute the detectors for specimens that already have a JSON")
    ap.add_argument("--refresh-clouds", action="store_true", help="also redo the GPU forward pass (ignore the npz cloud cache)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.n_sub: N_SUB = a.n_sub
    if a.zz_max_ck: ZZ_MAX_CK = a.zz_max_ck
    names = a.specimens if a.specimens else default_specimens()
    if not a.aggregate_only:
        for n in names:
            jf = os.path.join(OUT, f"nishino_flower_{n}.json")
            if os.path.exists(jf) and not a.force:
                print(f"skip {n} (results/open/nishino_flower_{n}.json exists)"); continue
            print(f"== {n} ==", flush=True)
            try:
                r = trajectory(n, every=a.every, zz_null=a.zz_null, reuse=not a.refresh_clouds)
                print(f"   done in {r['wall']}s", flush=True)
            except Exception:
                print(f"   FAILED {n}\n{traceback.format_exc()[-1500:]}", flush=True)
    aggregate()
