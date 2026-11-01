#!/usr/bin/env python3
"""O3.1 Hishi Akebono — T3 Pythia P7/P3 across all 154 public checkpoints (SPEC_O3.md; FINAL_SPEC §6.4 T3 row,
§6.8 T3 row, §8.2 O3.1). Both rows are *reported*, not pass/fail, and no cell can be `certified`: there is no
gpt-neox backend in whiteboxtopo.symmetry, so K-O-2 (re-compute on the gauge orbit) cannot be run — see SPEC_O3 §2.

  python hishi_akebono.py --model EleutherAI/pythia-410m --out hpcc/results/o31 --resume
  python hishi_akebono.py --model EleutherAI/pythia-70m --smoke --out hpcc/results/o31_smoke --device cuda
  python hishi_akebono.py --aggregate --out hpcc/results/o31

Reuse, never duplication: the EAP attribution graph and the revision list come from `hpcc/mejiro_ardan` (V7), the
pooled-residual reader from `hpcc/mejiro_ryan` (B5), the probe corpus and the BiasTop permutation test from
`experiments/B_bias/{blib,winning_ticket}` (B2/B3), the probe tokens from `experiments/K_darkmatter/eishin_flash` (K1),
and every instrument from `whiteboxtopo.{topology,spectral,graphs}`.

Resumable: one trajectory-CSV row per (model, revision) written as soon as it exists, `--resume` skips finished
revisions; the gauge-fixed clouds are cached in `clouds_<tag>.npz` so the trajectory-level instruments (zigzag,
eigenflow, change points, scorecard) re-run without a GPU. A time-out is re-submitted, never restarted.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, time, traceback
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE, os.path.join(ROOT, "hpcc"), os.path.join(ROOT, "experiments", "B_bias"),
          os.path.join(ROOT, "experiments", "K_darkmatter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mejiro_ardan as V7                                   # noqa: E402  EAP graph + the 154-revision list
from mejiro_ryan import pooled_resid                        # noqa: E402  B5's mean-pooled residual reader
import blib                                                 # noqa: E402  the B-block probe corpus
import winning_ticket as B                                  # noqa: E402  B2/B3: unit(), _biastop, _perm_test, _B3
from whiteboxtopo import graphs                             # noqa: E402
from whiteboxtopo.legibility.graph import symmetrise, threshold_density   # noqa: E402
from whiteboxtopo import topology as T                      # noqa: E402
from whiteboxtopo.topology.scaling import standardized_cloud, normalized_persistence   # noqa: E402
from whiteboxtopo.spectral import eigenflow, laplacian, smallest_eigs                  # noqa: E402
from whiteboxtopo.core.persistence import compute_diagrams  # noqa: E402
from whiteboxtopo.open import Scorecard, CellResult         # noqa: E402
from whiteboxtopo import symmetry as S                      # noqa: E402  AMENDMENT #O3-1: the gpt-neox backend
from whiteboxtopo.symmetry import neox as NEOX              # noqa: E402
from whiteboxtopo.symmetry.group import inverse_on_units    # noqa: E402

# ---- frozen constants (SPEC_O3.md; changes go to AMENDMENTS.md with a reason, never here) --------------------------
REVISIONS = V7.REVISIONS            # 154: step0, step{1,2,...,512}, step1000..step143000 by 1000
DENS = V7.DENS                      # 0.01 / 0.05 / 0.1 — the V7 edge densities
DENS_PRIMARY = 0.05                 # the Q_excess trace reported as P7-d
N_SEQ, SEQ_TOK = 16, 126            # the fixed 2 000-token OpenWebText-10k slice (V7 / K1)
N_CLOUD = 300                       # probe points in the activation cloud, the SAME 300 at every revision
N_PC = 20                           # fixed basis: top PCs of the FINAL revision's standardised cloud (Z1 convention)
KNN_K, N_EIGS, SMALL_THR = 15, 10, 0.1          # spectral.eigenflow (SPEC 2.10)
N_SUB = 100                         # zigzag point subsample, lowered by the budget ladder
SIMPLEX_BUDGET = 30_000             # O1.3 AMENDMENTS #6 (NOT the Z-block 800 k): Dionysus' cost grows with classes x positions
ZZ_MAX_CK, ZZ_MIN_CK, ZZ_MIN_LIFE = 40, 20, 3   # cloud-count cap / floor; "long" H1 bar (Z1)
ALPHA = 0.05                        # change-point significance
N_PERM = 2000                       # checkpoint-order-shuffle draws per change point
N_PERM_LABEL = 200                  # label-permutation draws for dbar
BIASTOP_NPERM = 100                 # B AMENDMENTS #1: the runner has always used 100, not the SPEC's 200
N_BIAS_FALLBACK = 200               # n per group when B5 has no verified requirement (largest grid point with 2n <= 480)
B2_KILL_SPLITS = 20                 # random halvings for the B2 kill test
DELTA_LOG = 0.5                     # onset agreement tolerance on the log10(1+step) axis (a factor of ~3.2 in steps)
SMOKE_REVISIONS = ["step0", "step1000", "step143000"]
# ---- K-O-2, AMENDMENT #O3-1 (whiteboxtopo.symmetry.neox now exists) ------------------------------------------------
N_GAUGE = 20                        # gauge samples per orbit revision on the cluster (--n-gauge; 0 = skip)
ORBIT_EVERY = 10                    # orbit revisions: every 10th of the 154, plus the final (pre-stated, cost bound)
ART_CV = 1e-3                       # SPEC_O K-O-2: CV > this downgrades the cell to "failed (artifact)"
GAUGE_KIND, GAUGE_ORTHOGONAL = "combined", False     # the full group: perms + GL(d_head) V/O + RoPE phases/scale + LN
# scalars K-O-2 certifies. Anything not in this list is evidence, not a certification criterion.
KO2_P7 = ["eigen_gap", "n_small", "np_H1", "Q", "treewidth_ub", "sparsity", "community_sizes", "zigzag_qualifying"]
KO2_P3 = ["dbar", "biastop_stat", "cos_mf_disjoint_halves", "cos_random_split_absmean", "g_resid_cos",
          "g_mlp_norm", "g_mlp_cos_aligned", "g_heads_cos_aligned"]
KO2_P7_REPAIRED = ["Q_pn", "treewidth_ub_pn", "sparsity_pn", "community_sizes_pn"]   # #O3-2: the repaired instrument
KO2_EVIDENCE_ONLY = {"g_heads_per_head_norm": "only an O(d_head) invariant, not a GL invariant (B AMENDMENTS #4); "
                                              "reported, never a certification criterion under orthogonal=False",
                     "*_pn": "the per-target-node EAP repair (SPEC_O3 #O3-2 / V AMENDMENTS #V7-1). Reported with its "
                             "own orbit CVs, but the P7 status is still driven by the PRE-REGISTERED instrument: "
                             "swapping which scalars gate a cell after seeing the result would be a threshold change"}

TRACES = ["eigen_gap", "n_small", "np_H1", "np_velocity", "Q_excess", "Q_excess_pn", "cos_to_final", "dbar",
          "biastop_stat", "zigzag_births"]
P7_INSTRUMENTS = {"P7-a zigzag": "zigzag_births", "P7-b eigenflow": "eigen_gap",
                  "P7-c np_velocity": "np_velocity", "P7-d legibility_drift": "Q_excess"}
NOT_REACHED = {
    "P0": ("the legibility numbers (Q, treewidth, sparsity) are computed here as the P7-d instrument, but the P0 gate "
           "verdict needs the V-block phase diagram and V8 found no calibrated structure->tractability map "
           "(legibility.profile.V8_CALIBRATION_ATTEMPT); issuing a verdict from the provisional rule would overclaim"),
    "P1": "FINAL_SPEC §8.2 puts the O3.1 row at P7/P3 only; the gpt-neox gauge backend now exists (AMENDMENT #O3-1) but no extraction/dictionary pass is budgeted in this row",
    "P2": "O2.1 Satono Crown owns P2 at T2; no path-product pass is budgeted in this row",
    "P4": "O1.4 Fine Motion (T1); the frontier P4 is K8 / gamma_cov (O4.1)",
    "P5": "O1.4 Fine Motion (T1)",
    "P6": "M-block memorization_signal per family; not this row",
}
GAUGE_PROV = ("gauge fixing (SPEC_O3 §2): the residual-stream cloud is invariant by construction under the head/"
              "MLP-unit permutations and within-head basis changes of G (they do not touch the residual stream); "
              "every residual coordinate is then divided by its own std over the probe at that revision "
              "(topology.scaling.standardized_cloud, the E2/Z5 repair), projected on the top-20 PCs of the FINAL "
              "revision's standardised cloud (frozen once), and centred + scaled to unit median pairwise distance per "
              "revision (Z AMENDMENTS #2). K-O-2 IS run (AMENDMENT #O3-1): whiteboxtopo.symmetry.neox supplies the "
              "GPTNeoXForCausalLM backend, so every certified scalar is recomputed on a gauge orbit "
              "(kind='combined', orthogonal=False: head perms, per-head GL(d_head) V/O, RoPE-plane phase+scale, GL on "
              "the non-rotary Q/K tail, MLP perms, LN gain/bias rescales, final LN) at a pre-stated subset of "
              "revisions; CV >= 1e-3 on any certified scalar downgrades the cell to 'failed (artifact)'.")


# ============================================================================================ small frozen utilities
def tag_of(model: str) -> str:
    return model.split("/")[-1]


def z_normalise(c):
    """Centre and scale to unit median pairwise distance (Z AMENDMENTS #2): shape across training, not size."""
    c = np.asarray(c, float) - np.asarray(c, float).mean(0)
    return c / max(float(np.median(pdist(c))), 1e-12)


def max_t(x):
    """Two-sample max-t change point: stat = max_i |mean(x[i:]) - mean(x[:i])| / (s_p sqrt(1/i + 1/(T-i))),
    index = the argmax split (first checkpoint of the post-change segment). Frozen, identical to O1.3."""
    x = np.asarray(x, float); Tn = len(x)
    if Tn < 6 or not np.isfinite(x).all() or float(np.nanstd(x)) <= 0:
        return 0.0, -1
    i = np.arange(1, Tn); cs = np.cumsum(x); cs2 = np.cumsum(x * x)
    n1 = i.astype(float); n2 = (Tn - i).astype(float)
    m1 = cs[i - 1] / n1; m2 = (cs[-1] - cs[i - 1]) / n2
    v1 = np.maximum(cs2[i - 1] / n1 - m1 ** 2, 0.0); v2 = np.maximum((cs2[-1] - cs2[i - 1]) / n2 - m2 ** 2, 0.0)
    sp = np.sqrt(np.maximum((n1 * v1 + n2 * v2) / max(Tn - 2, 1), 1e-300))
    t = np.abs(m2 - m1) / (sp * np.sqrt(1 / n1 + 1 / n2))
    j = int(np.nanargmax(t)); return float(t[j]), int(i[j])


def changepoint(x, steps, n_perm=N_PERM, seed=0):
    """max-t change point + checkpoint-ORDER-shuffle p-value + the half-rise onset (SPEC_O3 §5). `steps` is the
    training step of every revision, so the onset is reported in steps as well as in revision index."""
    x = np.asarray(x, float); steps = np.asarray(steps, float); n = len(x)
    out = dict(stat=0.0, p=1.0, cp_index=-1, onset_index=-1, onset_frac=None, onset_step=None,
               pre=None, post=None, n=int(n), detected=0, n_perm=int(n_perm))
    if n and not np.isfinite(x).all():                       # a trace with a hole is reported, not silently dropped
        out["note"] = f"{int(np.sum(~np.isfinite(x)))} non-finite entries"
        return out
    stat, i = max_t(x); out["stat"], out["cp_index"] = stat, i
    if i < 0:
        out["note"] = ("trace too short for the change-point rule (T < 6)" if n < 6 else "zero variance")
        return out
    rng = np.random.default_rng(seed)
    null = np.array([max_t(rng.permutation(x))[0] for _ in range(n_perm)])
    out["p"] = float((1 + np.sum(null >= stat)) / (n_perm + 1))
    pre, post = float(x[:i].mean()), float(x[i:].mean()); out["pre"], out["post"] = pre, post
    mid = 0.5 * (pre + post); sgn = 1.0 if post >= pre else -1.0
    hit = np.flatnonzero(sgn * (x[:i + 1] - mid) >= 0)
    j = int(hit[0]) if hit.size else i
    out["onset_index"] = j
    out["onset_frac"] = float(j / max(n - 1, 1))
    out["onset_step"] = float(steps[j])
    out["detected"] = int(out["p"] < ALPHA)
    out["rule"] = (f"two-sample max-t change point; p from a {n_perm}-draw checkpoint-order shuffle; detection iff "
                   f"p < {ALPHA}; onset = first crossing of the midpoint between the pre- and post-change means "
                   f"(half-rise, spectral.flow.crystallization_index convention)")
    return out


def log_step(s):
    return float(np.log10(1.0 + float(s)))


# ==================================================================================================== zigzag (P7-a)
def informative_scale(X):
    """Rips scale at which the final normalised cloud carries the most H1 classes (quantile 0.1 if none) —
    Z AMENDMENTS #2(b)."""
    D = pdist(X); dg = np.asarray(compute_diagrams(X, maxdim=1, backend="ripser", thresh=1.0).dgms[1]).reshape(-1, 2)
    if len(dg) == 0:
        return float(np.quantile(D, 0.1)), 0.1, 0
    grid = np.linspace(float(D.min()), 1.0, 200)
    b1 = np.array([np.sum((dg[:, 0] <= s) & (dg[:, 1] > s)) for s in grid])
    s = float(grid[int(np.argmax(b1))]); return s, float(np.mean(D <= s)), int(b1.max())


def estimate_simplices(sub, scale):
    """Cheap upper-bound screen before the authoritative dionysus count (O1.3 AMENDMENTS #6)."""
    worst = 0.0
    for i in range(len(sub) - 1):
        X = np.vstack([sub[i], sub[i + 1]]); n = len(X); dens = float((pdist(X) <= scale).mean())
        worst = max(worst, n + dens * n * (n - 1) / 2 + (dens ** 3) * n * (n - 1) * (n - 2) / 6)
    return worst * max(len(sub) - 1, 1)


def count_simplices(sub, scale, cap):
    import dionysus as d
    if estimate_simplices(sub, scale) > 4 * cap:
        return int(estimate_simplices(sub, scale)), True
    tot = 0
    for i in range(len(sub) - 1):
        tot += len(d.fill_rips(np.vstack([sub[i], sub[i + 1]]), 2, float(scale)))
        if tot > cap:
            return tot, True
    return tot, False


def zigzag_summary(clouds, steps, budget=SIMPLEX_BUDGET, n_sub0=N_SUB, max_ck=ZZ_MAX_CK, seed=0, verbose=True):
    """Zigzag over the revision clouds at the beta1-maximising scale under the simplex budget: lower the point
    subsample first (Z AMENDMENTS #2/#3 ladder), then halve the cloud count down to ZZ_MIN_CK. Returns the qualifying
    H1 classes (born after the first cloud used, alive at the last, lifetime >= ZZ_MIN_LIFE) and the per-revision
    birth counts on the FULL revision grid (zeros where a revision was not used)."""
    steps = np.asarray(steps, float); n_all = len(clouds)
    rng = np.random.default_rng(seed); pick = rng.choice(len(clouds[0]), min(n_sub0, len(clouds[0])), replace=False)
    levels = [min(max_ck, n_all)]
    while levels[-1] > ZZ_MIN_CK:
        levels.append(max(ZZ_MIN_CK, levels[-1] // 2))
    over, sub, ck_sel, scale, scale_q, b1max, n_simp, n_sub = True, None, None, 0.0, 0.0, 0, 0, n_sub0
    for n_ck in levels:
        ck_sel = np.unique(np.linspace(0, n_all - 1, min(n_ck, n_all)).round().astype(int))
        cs = [clouds[i] for i in ck_sel]
        for n_sub in (min(n_sub0, len(clouds[0])), 75, 50, 40, 30):
            sub = [c[pick[:n_sub]] for c in cs]
            scale, scale_q, b1max = informative_scale(sub[-1])
            t0 = time.time(); n_simp, over = count_simplices(sub, scale, budget)
            if verbose:
                print(f"    zigzag budget: n_ck {len(ck_sel)} n_sub {n_sub} scale {scale:.3f} q {scale_q:.3f} "
                      f"beta1max {b1max} simplices {n_simp} over={over} [{time.time() - t0:.0f}s]", flush=True)
            if not over:
                break
        if not over:
            break
    t0 = time.time(); z = T.zigzag(sub, scale, maxdim=1)
    if verbose:
        print(f"    zigzag done: {len(ck_sel)} clouds, {z.n_simplices} simplices [{time.time() - t0:.0f}s]", flush=True)
    bars = np.asarray(z.bars[1]).reshape(-1, 2)
    life = (np.where(np.isfinite(bars[:, 1]), bars[:, 1], z.n_positions) - bars[:, 0]) if len(bars) else np.zeros(0)
    ci = z.bars_in_cloud_index(1) if len(bars) else np.zeros((0, 2))
    last = len(ck_sel) - 1; qual = []
    for k in range(len(bars)):
        b_ck, d_ck = int(ci[k, 0]), int(ci[k, 1])
        if b_ck >= 1 and d_ck >= last and life[k] >= ZZ_MIN_LIFE:
            qual.append(dict(birth_ck=b_ck, birth_rev_index=int(ck_sel[b_ck]), birth_step=float(steps[ck_sel[b_ck]]),
                             life=float(life[k])))
    qual.sort(key=lambda r: -r["life"])
    births_full = np.zeros(n_all, int)
    for q in qual:
        births_full[q["birth_rev_index"]] += 1
    births_sel = np.zeros(len(ck_sel), int)
    for q in qual:
        births_sel[q["birth_ck"]] += 1
    return dict(scale=float(scale), scale_quantile=float(scale_q), beta1_final_max=int(b1max), n_sub=int(n_sub),
                budget_reduced=int(n_sub < min(n_sub0, len(clouds[0]))), over_budget=int(over),
                n_clouds=int(len(ck_sel)), ck_sel=ck_sel.tolist(), clouds_reduced=int(len(ck_sel) < n_all),
                resolution_frac=float(1.0 / max(len(ck_sel) - 1, 1)), n_simplices=int(z.n_simplices),
                n_bars_H1=int(len(bars)), n_bars_H0=int(len(np.asarray(z.bars[0]).reshape(-1, 2))),
                max_life_not_initial=float(max([life[k] for k in range(len(bars)) if ci[k, 0] >= 1], default=0.0)),
                qualifying=qual, births_per_ck=births_sel.tolist(), births_full=births_full.tolist(),
                betti1_curve=np.asarray(z.betti_curve(1))[::2].tolist(),
                betti0_curve=np.asarray(z.betti_curve(0))[::2].tolist())


def zigzag_order_null(clouds, ck_sel, scale, n_sub, n_perm, seed=0):
    """Cloud-ORDER permutation null for the zigzag: same clouds, no time structure; statistic = the longest
    non-initially-born H1 bar. n_perm = 0 -> not run (each draw is one full zigzag)."""
    if n_perm <= 0:
        return dict(n_perm=0, note="no permutation null run (--zz-null 0): each draw is one full zigzag; the zigzag "
                                   "evidence is descriptive only")
    rng = np.random.default_rng(seed); pick = rng.choice(len(clouds[0]), min(n_sub, len(clouds[0])), replace=False)
    sub = [clouds[i][pick] for i in ck_sel]; vals = []
    for b in range(n_perm):
        order = np.random.default_rng(1000 + b).permutation(len(sub))
        z = T.zigzag([sub[i] for i in order], scale, maxdim=1)
        bars = np.asarray(z.bars[1]).reshape(-1, 2)
        if not len(bars):
            vals.append(0.0); continue
        life = np.where(np.isfinite(bars[:, 1]), bars[:, 1], z.n_positions) - bars[:, 0]
        ci = z.bars_in_cloud_index(1)
        vals.append(float(max([life[k] for k in range(len(bars)) if ci[k, 0] >= 1], default=0.0)))
    return dict(n_perm=int(n_perm), values=vals, mean=float(np.mean(vals)), max=float(np.max(vals)))


# ============================================================================================== per-revision passes
@torch.no_grad()
def resid_cloud(model, Tt, mid, idx, batch=4):
    """Residual stream at the OUTPUT of layer `mid`, one point per token, at the fixed probe positions `idx`."""
    out = []
    for i in range(0, len(Tt), batch):
        hs = model(Tt[i:i + batch], output_hidden_states=True).hidden_states[mid + 1]
        out.append(hs[:, 1:].reshape(-1, hs.shape[-1]).float().cpu().numpy())
    A = np.concatenate(out)
    return A[idx]


EAP_VARIANTS = (("pre_registered", "eap_graph"), ("per_node", "eap_graph_per_node"))


def legibility_rows(model, Tt, mid, dev):
    """P7-d: the V7 rows — Leiden Q, degree-preserving-rewired null, treewidth bound, sparsity, at 3 densities.
    BOTH graphs are built (SPEC_O3 #O3-2): `mejiro_ardan.eap_graph` (the PRE-REGISTERED instrument, one gradient per
    layer) and `mejiro_ardan.eap_graph_per_node` (the repair: one gradient per target node). Imported, never
    re-implemented. Rows carry `graph` = pre_registered | per_node."""
    layers = list(range(mid - 2, mid + 3)); rows = []
    for label, fn in EAP_VARIANTS:
        E, nodes = getattr(V7, fn)(model, Tt, layers, dev)
        A_full = symmetrise(E)
        for dens in DENS:
            A = threshold_density(A_full, dens)
            Q, memb = graphs.modularity(A); q0, sd = graphs.rewired_null(A, n=20)
            rows.append(dict(graph=label, density=dens, n_nodes=len(nodes), n_edges=int((A != 0).sum() // 2),
                             n_distinct_edge_values=int(len(np.unique(np.abs(A_full)[np.triu_indices(len(nodes), 1)]))),
                             Q=float(Q), Q_null=float(q0), Q_null_sd=float(sd), Q_excess=float(Q - q0),
                             n_communities=int(len(np.unique(memb))), treewidth_ub=int(graphs.treewidth_ub(A)),
                             sparsity=float(graphs.sparsity(A_full))))
    return rows


def bias_row(model, tok, mid, dev, male, female, g_final, n_bias, biastop_nperm, label_null, seed=0):
    """P3: the B2 direction + kill test and the B3 BiasTop permutation test at one revision."""
    M = pooled_resid(model, tok, male, mid, dev); F = pooled_resid(model, tok, female, mid, dev)
    allX = np.vstack([M, F]); g = B.unit(M.mean(0) - F.mean(0))
    sd = float(allX.std(0).mean())
    dbar = float(np.linalg.norm(M.mean(0) - F.mean(0)) / max(sd, 1e-12))
    rng = np.random.default_rng(seed)
    # --- B2 kill test (B SPEC §B2), recomputed at THIS revision -----------------------------------------------
    half = min(len(M), len(F)) // 2
    pm = rng.permutation(len(M)); pf = rng.permutation(len(F))
    g_a = B.unit(M[pm[:half]].mean(0) - F[pf[:half]].mean(0))
    g_b = B.unit(M[pm[half:2 * half]].mean(0) - F[pf[half:2 * half]].mean(0))
    cos_mf = float(g_a @ g_b)
    rnd = []
    for s in range(B2_KILL_SPLITS):
        r = np.random.default_rng(1000 + s).permutation(len(allX)); q = len(allX) // 4
        u = B.unit(allX[r[:q]].mean(0) - allX[r[q:2 * q]].mean(0))
        v = B.unit(allX[r[2 * q:3 * q]].mean(0) - allX[r[3 * q:4 * q]].mean(0))
        rnd.append(float(u @ v))
    cos_rand_abs = float(np.mean(np.abs(rnd)))
    # --- label-permutation null for dbar ----------------------------------------------------------------------
    p_dbar = None
    if label_null > 0:
        nm = len(M); vals = []
        for b in range(label_null):
            r = np.random.default_rng(5000 + b).permutation(len(allX))
            vals.append(float(np.linalg.norm(allX[r[:nm]].mean(0) - allX[r[nm:]].mean(0)) / max(sd, 1e-12)))
        p_dbar = float((1 + np.sum(np.asarray(vals) >= dbar)) / (label_null + 1))
    # --- B3 BiasTop: W2 between Rips-H1 diagrams of the PCA-20 clouds; _perm_test IS a label-permutation null ---
    mu = allX.mean(0); P20 = np.linalg.svd(allX - mu, full_matrices=False)[2][:20].T
    B._B3.update(M=M, F=F, mu=mu, P20=P20, g=g, sd=sd)
    n = int(min(n_bias, len(M), len(F)))
    ia = np.random.default_rng(7).choice(len(M), n, replace=False)
    ib = np.random.default_rng(8).choice(len(F), n, replace=False)
    A_, B_ = M[ia], F[ib]
    stat = float(B._biastop(A_, B_))
    p_bt = float(B._perm_test(A_, B_, np.random.default_rng(9), n_perm=biastop_nperm))
    return dict(dbar=dbar, cos_to_final=float(g @ g_final), biastop_stat=stat, biastop_p=p_bt, biastop_n=n,
                biastop_nperm=int(biastop_nperm), cos_mf_disjoint_halves=cos_mf, cos_random_split_absmean=cos_rand_abs,
                b2_kill_pass=int(cos_mf > 0.9 and cos_rand_abs < 0.5), p_dbar_label=p_dbar, n_male=len(M),
                n_female=len(F)), g


# ================================================================== K-O-2: the gauge orbit (AMENDMENT #O3-1)
@torch.no_grad()
def pooled_neox(model, tok, sents, mid, dev, bs=32):
    """Mean-pooled (residual after layer `mid`, post-GELU MLP hidden of `mid`, per-head outputs z of `mid`) —
    the three coordinate systems B2 audits, for GPT-NeoX. Mirrors winning_ticket.pooled (GPT-2)."""
    blk = NEOX._blocks(model)[mid]
    cap = {}
    h1 = blk.mlp.act.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp", o.detach()))
    h2 = blk.attention.dense.register_forward_pre_hook(lambda m, i: cap.__setitem__("z", i[0].detach()))
    R, M, Z = [], [], []
    try:
        for i in range(0, len(sents), bs):
            enc = tok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=40).to(dev)
            out = model(**enc, output_hidden_states=True)
            mk = enc["attention_mask"].unsqueeze(-1).float()
            def mp(x):
                return ((x.float() * mk).sum(1) / mk.sum(1)).cpu().numpy()
            R.append(mp(out.hidden_states[mid + 1])); M.append(mp(cap["mlp"])); Z.append(mp(cap["z"]))
    finally:
        h1.remove(); h2.remove()
    c = NEOX._cfg(model)
    return np.concatenate(R), np.concatenate(M), np.concatenate(Z).reshape(-1, c["H"], c["dh"])


def single_spectrum(Z):
    """(spectral gap, n_small) of ONE gauge-fixed cloud — the per-checkpoint quantities eigenflow reports."""
    L, _ = laplacian(np.asarray(Z, float), k=min(KNN_K, len(Z) - 1))
    lam, _ = smallest_eigs(L, min(N_EIGS, L.shape[0])); lam = np.sort(lam)
    return float(lam[1] - lam[0]) if len(lam) >= 2 else 0.0, int((lam < SMALL_THR).sum())


def align_mlp(g, mid, v):
    """Undo a renaming on an MLP-hidden-coordinate vector: un-permute back to base unit order. No scalings exist
    on this axis (GELU is not homogeneous, so the backend never samples them)."""
    return inverse_on_units(g, ("mlp", mid), np.asarray(v, float), axis=0)


def align_heads(g, mid, V):
    """Undo a renaming on a per-head-output array (H, d_head). `neox.apply` rotates each head at its ORIGINAL index
    and THEN permutes heads (B AMENDMENTS #4's ordering trap), so un-permute FIRST, then left-multiply by B_i^{-1}
    at the base index i (z_j^new = B_{p[j]} z_{p[j]}^base)."""
    V = inverse_on_units(g, ("head", mid), np.asarray(V, float), axis=0)
    Bs = g.rots.get(("vo", mid))
    if Bs is not None:
        V = np.stack([np.linalg.solve(np.asarray(Bs[i], float), V[i]) for i in range(len(V))])
    return V


def orbit_measure(model, Tt, mid, idx, mu, P20, tok, male, female, dev, n_bias, biastop_nperm):
    """Every K-O-2 quantity, computed on whatever model is handed in (base or renamed). One forward pass family."""
    A = resid_cloud(model, Tt, mid, idx)
    Z = z_normalise((standardized_cloud(A, how="std") - mu) @ P20)
    gap, nsm = single_spectrum(Z)
    npz = normalized_persistence(A, maxdim=1, how="std")
    layers = list(range(mid - 2, mid + 3)); leg = {}
    for label, fn in EAP_VARIANTS:                     # #O3-2: the pre-registered graph AND the per-node repair
        E, nodes = getattr(V7, fn)(model, Tt, layers, dev)
        Af = symmetrise(E); Agr = threshold_density(Af, DENS_PRIMARY); Q, memb = graphs.modularity(Agr)
        sfx = "" if label == "pre_registered" else "_pn"
        leg[f"Q{sfx}"] = float(Q); leg[f"treewidth_ub{sfx}"] = int(graphs.treewidth_ub(Agr))
        leg[f"sparsity{sfx}"] = float(graphs.sparsity(Af))
        leg[f"community_sizes{sfx}"] = sorted(np.bincount(np.asarray(memb)).tolist(), reverse=True)
        leg[f"node_roles{sfx}"] = sorted([n_[1] for n_ in nodes])
        leg[f"n_distinct_edge_values{sfx}"] = int(len(np.unique(np.abs(Af)[np.triu_indices(len(nodes), 1)])))
    R, Mh, Hd = pooled_neox(model, tok, male + female, mid, dev)
    nM = len(male)
    g_res = B.unit(R[:nM].mean(0) - R[nM:].mean(0))
    g_mlp = Mh[:nM].mean(0) - Mh[nM:].mean(0)
    g_hd = Hd[:nM].mean(0) - Hd[nM:].mean(0)
    allX = np.vstack([R[:nM], R[nM:]]); sd = float(allX.std(0).mean())
    dbar = float(np.linalg.norm(R[:nM].mean(0) - R[nM:].mean(0)) / max(sd, 1e-12))
    half = min(nM, len(R) - nM) // 2
    pm = np.random.default_rng(0).permutation(nM); pf = np.random.default_rng(1).permutation(len(R) - nM)
    ga = B.unit(R[:nM][pm[:half]].mean(0) - R[nM:][pf[:half]].mean(0))
    gb = B.unit(R[:nM][pm[half:2 * half]].mean(0) - R[nM:][pf[half:2 * half]].mean(0))
    rnd = []
    for s in range(B2_KILL_SPLITS):
        r = np.random.default_rng(1000 + s).permutation(len(allX)); q = len(allX) // 4
        rnd.append(float(B.unit(allX[r[:q]].mean(0) - allX[r[q:2 * q]].mean(0)) @
                         B.unit(allX[r[2 * q:3 * q]].mean(0) - allX[r[3 * q:4 * q]].mean(0))))
    mu_b = allX.mean(0); P20b = np.linalg.svd(allX - mu_b, full_matrices=False)[2][:20].T
    B._B3.update(M=R[:nM], F=R[nM:], mu=mu_b, P20=P20b, g=g_res, sd=sd)
    n = int(min(n_bias, nM, len(R) - nM))
    A_ = R[:nM][np.random.default_rng(7).choice(nM, n, replace=False)]
    B_ = R[nM:][np.random.default_rng(8).choice(len(R) - nM, n, replace=False)]
    return dict(cloud=Z.astype(np.float32), eigen_gap=gap, n_small=nsm,
                np_H1=float(npz["total_persistence_H1"]), **leg,
                dbar=dbar, biastop_stat=float(B._biastop(A_, B_)),
                cos_mf_disjoint_halves=float(ga @ gb), cos_random_split_absmean=float(np.mean(np.abs(rnd))),
                g_resid=g_res, g_mlp=g_mlp, g_heads=g_hd,
                g_heads_per_head_norm=np.linalg.norm(g_hd, axis=1))


def ko2_status(produced, ver, artifacts):
    """Cell status under SPEC_O K-O-2 (AMENDMENT #O3-1): certified iff the instruments produced values AND the gauge
    orbit was run AND no certified scalar moved by CV >= ART_CV; 'failed' (artifact) if one did; 'partial' if the
    orbit was not run (gauge-fixed but not gauge-certified)."""
    if not produced:
        return "not_reached"
    if not ver.get("ran"):
        return "partial"
    return "failed" if artifacts else "certified"


def cv_of(vals):
    v = np.asarray(vals, float)
    if v.ndim > 1:
        return float(np.max(v.std(0) / (np.abs(v).mean(0) + 1e-12)))
    return float(v.std() / (abs(v.mean()) + 1e-12))


def ko2(model, model_id, rev, Tt, mid, idx, mu, P20, tok, male, female, dev, n_bias, biastop_nperm, n_gauge,
        seed=0, verbose=True):
    """SPEC_O K-O-2 on one revision: recompute every certified scalar on `n_gauge` random elements of G (the full
    combined group, orthogonal=False) and report the CV. `whiteboxtopo.symmetry.neox` supplies the backend."""
    t0 = time.time()
    base = orbit_measure(model, Tt, mid, idx, mu, P20, tok, male, female, dev, n_bias, biastop_nperm)
    rng = np.random.default_rng(seed)
    per = {k: [base[k]] for k in ("eigen_gap", "n_small", "np_H1", "Q", "treewidth_ub", "sparsity", "dbar",
                                  "biastop_stat", "cos_mf_disjoint_halves", "cos_random_split_absmean",
                                  "Q_pn", "treewidth_ub_pn", "sparsity_pn", "n_distinct_edge_values",
                                  "n_distinct_edge_values_pn")}
    per["g_heads_per_head_norm"] = [base["g_heads_per_head_norm"]]
    sets = {k: [base[k]] for k in ("community_sizes", "node_roles", "community_sizes_pn", "node_roles_pn")}
    cosines = dict(g_resid_cos=[1.0], g_mlp_cos_aligned=[1.0], g_heads_cos_aligned=[1.0],
                   g_mlp_norm=[float(np.linalg.norm(base["g_mlp"]))])
    clouds, devs = [base["cloud"]], []
    for j in range(n_gauge):
        g = NEOX.sample(model, rng, kind=GAUGE_KIND, orthogonal=GAUGE_ORTHOGONAL)
        gm = S.apply(model, g)
        try:
            tk = Tt[:1, :32].detach().cpu().numpy()
            y0 = NEOX.forward(model, tk); y1 = NEOX.forward(gm, tk)
            devs.append(float(np.max(np.abs(y1 - y0)) / max(float(np.abs(y0).mean()), 1e-12)))
            m = orbit_measure(gm, Tt, mid, idx, mu, P20, tok, male, female, dev, n_bias, biastop_nperm)
            for k in per:
                per[k].append(m[k])
            for k in sets:
                sets[k].append(m[k])
            cosines["g_resid_cos"].append(float(m["g_resid"] @ base["g_resid"]))
            am = align_mlp(g, mid, m["g_mlp"]); ah = align_heads(g, mid, m["g_heads"])
            cosines["g_mlp_cos_aligned"].append(float(B.unit(am) @ B.unit(base["g_mlp"])))
            cosines["g_heads_cos_aligned"].append(float(B.unit(ah.ravel()) @ B.unit(base["g_heads"].ravel())))
            cosines["g_mlp_norm"].append(float(np.linalg.norm(m["g_mlp"])))
            clouds.append(m["cloud"])
        finally:
            del gm
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
    cvs = {k: cv_of(v) for k, v in per.items()}
    cvs.update({k: cv_of(v) for k, v in cosines.items()})
    for k, v in sets.items():
        cvs[k] = 0.0 if all(x == v[0] for x in v) else float("inf")
    out = dict(revision=rev, n_gauge=int(n_gauge), kind=GAUGE_KIND, orthogonal=GAUGE_ORTHOGONAL, device=str(dev),
               torch_version=str(torch.__version__),
               cv=cvs, values={k: (np.asarray(v).tolist() if k != "g_heads_per_head_norm" else None)
                               for k, v in per.items()},
               cosines={k: v for k, v in cosines.items()},
               sets_identical={k: bool(all(x == v[0] for x in v)) for k, v in sets.items()},
               output_deviation_rel=dict(max=float(np.max(devs)) if devs else None,
                                         mean=float(np.mean(devs)) if devs else None,
                                         note="max |logits_g - logits| / mean|logits| on 32 probe tokens; this IS the "
                                              "fp32 floor and is the tolerance below which two quantities are "
                                              "'identical'"),
               evidence_only=KO2_EVIDENCE_ONLY, art_cv=ART_CV, wall_s=round(time.time() - t0, 1))
    if verbose:
        print(f"    K-O-2 {rev}: n_gauge {n_gauge} fp32 floor {out['output_deviation_rel']['max']:.2e} "
              f"maxCV {max(cvs.values()):.2e} [{out['wall_s']}s]", flush=True)
    return out, np.stack(clouds)


def orbit_indices(n, every=ORBIT_EVERY):
    """Pre-stated K-O-2 revision subset: every `every`-th revision plus the final one."""
    return sorted(set(list(range(0, n, every)) + [n - 1]))


# =========================================================================================== basis + model handling
def load_model(model_id, rev, dev):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, revision=rev, torch_dtype=torch.float32).to(dev).eval()


def purge_revision(model_id, rev):
    """Free the HF snapshot of one revision (154 x ~0.9 GB for 410m / ~2 GB for 1b would otherwise stay on disk).
    Never run concurrently with V7 on the same HF_HOME."""
    try:
        from huggingface_hub import scan_cache_dir
        c = scan_cache_dir()
        for repo in c.repos:
            if repo.repo_id == model_id:
                for r in repo.revisions:
                    if rev in r.refs:
                        c.delete_revisions(r.commit_hash).execute(); return True
    except Exception as e:
        print(f"  purge failed ({type(e).__name__}: {e})", flush=True)
    return False


def ensure_basis(model_id, out, revs, n_seq, n_cloud, dev, male, female, purge=False):
    """Freeze, from the LAST revision in the list: the mid layer, the fixed probe-point indices, the standardised
    cloud's mean and top-20 PCs, and the final bias direction g_final. Cached; computed once per model."""
    tag = tag_of(model_id); p = os.path.join(out, f"basis_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return dict(mid=int(z["mid"]), idx=z["idx"], mu=z["mu"], P20=z["P20"], g_final=z["g_final"],
                    rev=str(z["rev"]), n_tokens=int(z["n_tokens"]))
    from transformers import AutoTokenizer
    import eishin_flash
    tok = AutoTokenizer.from_pretrained(model_id); eishin_flash.CACHE = out
    toks = eishin_flash.load_tokens(tok)[:n_seq, :SEQ_TOK]
    Tt = torch.tensor(toks, device=dev)
    rev = revs[-1]
    print(f"[basis] {tag}: loading {rev} to freeze the basis", flush=True)
    model = load_model(model_id, rev, dev)
    mid = model.config.num_hidden_layers // 2
    n_pts = toks.shape[0] * (SEQ_TOK - 1)
    idx = np.sort(np.random.default_rng(0).choice(n_pts, min(n_cloud, n_pts), replace=False))
    A = resid_cloud(model, Tt, mid, idx)
    S = standardized_cloud(A, how="std")
    mu = S.mean(0); P20 = np.linalg.svd(S - mu, full_matrices=False)[2][:N_PC].T
    Mf = pooled_resid(model, tok, male, mid, dev); Ff = pooled_resid(model, tok, female, mid, dev)
    g_final = B.unit(Mf.mean(0) - Ff.mean(0))
    np.savez(p, mid=mid, idx=idx, mu=mu, P20=P20, g_final=g_final, rev=rev, n_tokens=int(toks.size))
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    if purge:
        purge_revision(model_id, rev)
    return dict(mid=int(mid), idx=idx, mu=mu, P20=P20, g_final=g_final, rev=rev, n_tokens=int(toks.size))


# ================================================================================================== the revision run
def run_model(a, model_id, revs):
    tag = tag_of(model_id); out = a.out; os.makedirs(out, exist_ok=True)
    traj_p = os.path.join(out, f"O31_{tag}_trajectory.csv")
    leg_p = os.path.join(out, f"O31_{tag}_legibility.csv")
    cl_p = os.path.join(out, f"clouds_{tag}.npz")
    male, female = blib.build_corpus_unique(0)
    if a.bias_n_sent:
        male, female = male[:a.bias_n_sent], female[:a.bias_n_sent]
    bas = ensure_basis(model_id, out, revs, a.n_seq, a.n_cloud, a.device, male, female, purge=a.purge)
    from transformers import AutoTokenizer
    import eishin_flash
    tok = AutoTokenizer.from_pretrained(model_id); eishin_flash.CACHE = out
    toks = eishin_flash.load_tokens(tok)[:a.n_seq, :SEQ_TOK]
    Tt = torch.tensor(toks, device=a.device)
    mid, idx, mu, P20, g_final = bas["mid"], bas["idx"], bas["mu"], bas["P20"], bas["g_final"]

    ko2_p = os.path.join(out, f"O31_{tag}_ko2.json")
    orb_p = os.path.join(out, f"orbit_{tag}.npz")
    rows = pd.read_csv(traj_p).to_dict("records") if (a.resume and os.path.exists(traj_p)) else []
    legs = pd.read_csv(leg_p).to_dict("records") if (a.resume and os.path.exists(leg_p)) else []
    clouds = dict(np.load(cl_p)) if (a.resume and os.path.exists(cl_p)) else {}
    ko2s = json.load(open(ko2_p)) if (a.resume and os.path.exists(ko2_p)) else {}
    orbs = dict(np.load(orb_p)) if (a.resume and os.path.exists(orb_p)) else {}
    done = {r["revision"] for r in rows} & set(clouds)
    n_bias = a.bias_n if a.bias_n else b5_requirement(model_id)
    orbit = {revs[i] for i in orbit_indices(len(revs), a.orbit_every)} if a.n_gauge > 0 else set()

    for k, rev in enumerate(revs):
        if rev in done and (rev not in orbit or rev in ko2s):
            continue
        t0 = time.time()
        try:
            model = load_model(model_id, rev, a.device)
        except Exception as e:
            print(f"skip {rev}: {type(e).__name__}: {e}", flush=True); continue
        try:
            if rev in orbit and rev not in ko2s:
                try:            # a K-O-2 failure must never cost the trajectory row (negatives are deliverables)
                    ko2s[rev], orbs[rev] = ko2(model, model_id, rev, Tt, mid, idx, mu, P20, tok, male, female,
                                               a.device, n_bias, a.biastop_nperm, a.n_gauge, verbose=not a.quiet)
                    json.dump(ko2s, open(ko2_p, "w"), indent=1, default=str)
                    np.savez_compressed(orb_p, **orbs)
                except Exception as e:
                    print(f"K-O-2 FAIL {rev}: {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}", flush=True)
            if rev in done:
                continue
            A = resid_cloud(model, Tt, mid, idx)
            Y = (standardized_cloud(A, how="std") - mu) @ P20
            Z = z_normalise(Y)
            clouds[rev] = Z.astype(np.float32)
            npz = normalized_persistence(A, maxdim=1, how="std")                     # Z5, on the full-dim cloud
            bias, _ = bias_row(model, tok, mid, a.device, male, female, g_final, n_bias, a.biastop_nperm,
                               a.label_null)
            lr = legibility_rows(model, Tt, mid, a.device)
            for r in lr:
                legs.append(dict(model=model_id, revision=rev, step=int(rev[4:]), **r))
            prim = [r for r in lr if abs(r["density"] - DENS_PRIMARY) < 1e-9 and r["graph"] == "pre_registered"][0]
            prim_pn = [r for r in lr if abs(r["density"] - DENS_PRIMARY) < 1e-9 and r["graph"] == "per_node"][0]
            row = dict(model=model_id, revision=rev, step=int(rev[4:]), rev_index=k, mid_layer=mid,
                       n_cloud=int(len(Z)), np_H1=npz["total_persistence_H1"], np_H0=npz["total_persistence_H0"],
                       np_bars_H1=npz["n_bars_H1"], np_entropy_H1=npz["persistent_entropy_H1"],
                       Q=prim["Q"], Q_null=prim["Q_null"], Q_excess=prim["Q_excess"],
                       treewidth_ub=prim["treewidth_ub"], sparsity=prim["sparsity"],
                       n_communities=prim["n_communities"],
                       Q_pn=prim_pn["Q"], Q_null_pn=prim_pn["Q_null"], Q_excess_pn=prim_pn["Q_excess"],
                       treewidth_ub_pn=prim_pn["treewidth_ub"], sparsity_pn=prim_pn["sparsity"],
                       n_distinct_edge_values=prim["n_distinct_edge_values"],
                       n_distinct_edge_values_pn=prim_pn["n_distinct_edge_values"],
                       **bias, wall_s=round(time.time() - t0, 1))
            rows.append(row)
            pd.DataFrame(rows).to_csv(traj_p, index=False)
            pd.DataFrame(legs).to_csv(leg_p, index=False)
            np.savez_compressed(cl_p, **clouds)
            print(f"[{tag}] {rev:12s} np_H1 {row['np_H1']:.4g} Q_exc {row['Q_excess']:+.4f} cos_fin "
                  f"{row['cos_to_final']:+.3f} dbar {row['dbar']:.3f} biastop p {row['biastop_p']:.3f} "
                  f"b2_kill {row['b2_kill_pass']} [{row['wall_s']}s]", flush=True)
        except Exception as e:
            print(f"FAIL {rev}: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        finally:
            del model
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if a.purge:
                purge_revision(model_id, rev)
    return finalise(a, model_id, traj_p, cl_p)


def b5_requirement(model_id):
    """n per group from B5 (hpcc/mejiro_ryan). Absent -> the fallback n, and the cell is power-UNVERIFIED."""
    for p in (os.path.join(ROOT, "hpcc", "results", "b5", "B5_requirements.json"),
              os.path.join(ROOT, "hpcc", "results", "smoke_local", "b5", "B5_requirements.json")):
        if os.path.exists(p):
            try:
                for r in json.load(open(p)):
                    if r.get("model") == model_id and r.get("n_required_power80_stretch05"):
                        return int(r["n_required_power80_stretch05"])
            except Exception:
                pass
    return N_BIAS_FALLBACK


def b5_power_verified(model_id):
    return b5_requirement(model_id) != N_BIAS_FALLBACK or False


# ================================================================================ trajectory instruments + scorecard
def finalise(a, model_id, traj_p, cl_p):
    """Everything that needs the whole trajectory: eigenflow, the zigzag, the change points and the Scorecard.
    Runs off the CSV + the cloud cache; no GPU."""
    tag = tag_of(model_id)
    df = pd.read_csv(traj_p).sort_values("step").reset_index(drop=True)
    cl = np.load(cl_p)
    revs = [r for r in df.revision if r in cl.files]
    df = df[df.revision.isin(revs)].reset_index(drop=True)
    clouds = [cl[r].astype(np.float64) for r in df.revision]
    steps = df.step.to_numpy(float)
    ev = dict(model=model_id, n_revisions=int(len(df)), mid_layer=int(df.mid_layer.iloc[0]),
              revisions=[str(r) for r in df.revision], steps=steps.tolist(), gauge=GAUGE_PROV, instruments={},
              nulls={}, spec="experiments/O_opener/SPEC_O3.md")

    # ---- P7-b eigenflow ------------------------------------------------------------------------------------
    try:
        ef = eigenflow(clouds, k=min(KNN_K, len(clouds[0]) - 1), n_eigs=N_EIGS, threshold=SMALL_THR)
        df["eigen_gap"] = ef.gap; df["n_small"] = ef.n_small
        ev["instruments"]["eigenflow"] = dict(k=int(ef.k), n_eigs=int(ef.n_eigs), threshold=float(ef.threshold),
                                              gap=ef.gap.tolist(), n_small=ef.n_small.astype(int).tolist(),
                                              crystallization_index=int(ef.crystallization_index()))
    except Exception as e:
        df["eigen_gap"] = np.nan; df["n_small"] = np.nan
        ev["instruments"]["eigenflow"] = dict(error=f"{type(e).__name__}: {e}")

    # ---- P7-c normalised-persistence velocity (Z5) ----------------------------------------------------------
    np_H1 = df.np_H1.to_numpy(float)
    vel = np.concatenate([[0.0], np.diff(np_H1)])
    dstep = np.concatenate([[1.0], np.maximum(np.diff(steps), 1.0)])
    df["np_velocity"] = vel; df["np_velocity_per_step"] = vel / dstep
    ev["instruments"]["np_velocity"] = dict(definition="NP(t) = total persistence H1 of the standardised cloud "
                                            "(topology.normalized_persistence, how='std'); velocity = NP(t) - NP(t-1) "
                                            "in revision-index units; a step-normalised copy is reported alongside",
                                            np_H1=np_H1.tolist(), velocity=vel.tolist())

    # ---- P7-a zigzag ---------------------------------------------------------------------------------------
    try:
        if len(clouds) < 2:
            raise ValueError(f"zigzag needs >= 2 checkpoint clouds, have {len(clouds)}")
        zz = zigzag_summary(clouds, steps, seed=0, verbose=not a.quiet)
        zz["null_order_permutation"] = zigzag_order_null(clouds, zz["ck_sel"], zz["scale"], zz["n_sub"], a.zz_null)
        df["zigzag_births"] = np.asarray(zz["births_full"], float)[:len(df)]
        ev["instruments"]["zigzag"] = zz
    except Exception as e:
        df["zigzag_births"] = np.nan
        ev["instruments"]["zigzag"] = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-700:])

    # ---- change points on every trace ----------------------------------------------------------------------
    cps = {}
    for name in TRACES:
        if name not in df.columns:
            cps[name] = dict(note="trace not produced"); continue
        x = df[name].to_numpy(float)
        if not np.isfinite(x).all() or np.nanstd(x) == 0:
            cps[name] = dict(note="trace constant or incomplete", n=int(len(x)),
                             n_finite=int(np.sum(np.isfinite(x)))); continue
        cps[name] = changepoint(x, steps, n_perm=a.n_perm, seed=0)
    ev["change_points"] = cps
    ev["nulls"] = dict(
        checkpoint_order_shuffle=f"{a.n_perm} draws, every change point (detecting or not)",
        label_permutation=(f"{a.label_null} draws for dbar at every revision; the BiasTop p-value is itself a "
                           f"{a.biastop_nperm}-draw label permutation (winning_ticket._perm_test)"),
        degree_preserving_rewiring="20 draws per revision per density (graphs.rewired_null) -> Q_null, Q_excess",
        zigzag_cloud_order=ev["instruments"].get("zigzag", {}).get("null_order_permutation", {}))

    # ---- onset agreement across the four P7 instruments -----------------------------------------------------
    det = {k: cps[v] for k, v in P7_INSTRUMENTS.items()
           if isinstance(cps.get(v), dict) and cps[v].get("detected") and cps[v].get("onset_step") is not None}
    names = sorted(det); deltas = {}
    for i, u in enumerate(names):
        for w in names[i + 1:]:
            deltas[f"{u} vs {w}"] = round(abs(log_step(det[u]["onset_step"]) - log_step(det[w]["onset_step"])), 4)
    agree = dict(n_detecting=len(names), detecting=names, pairwise_delta_log10step=deltas,
                 max_pairwise_delta=(max(deltas.values()) if deltas else None),
                 concur=(bool(deltas) and max(deltas.values()) <= DELTA_LOG),
                 tolerance=DELTA_LOG,
                 onsets_step={k: det[k]["onset_step"] for k in names})
    ev["p7_agreement"] = agree

    # ---- K-O-2: the gauge orbit (AMENDMENT #O3-1) -----------------------------------------------------------
    ko2_p = os.path.join(a.out, f"O31_{tag}_ko2.json"); orb_p = os.path.join(a.out, f"orbit_{tag}.npz")
    ver = dict(ran=False, reason="no K-O-2 pass on disk (--n-gauge 0, or the orbit revisions were not reached)")
    if os.path.exists(ko2_p):
        ko2s = json.load(open(ko2_p))
        orbit_revs = [r for r in df.revision if r in ko2s]
        if orbit_revs:
            cvmax = {}
            for r in orbit_revs:
                for k, v in ko2s[r]["cv"].items():
                    cvmax[k] = max(cvmax.get(k, 0.0), float(v))
            floors = [ko2s[r]["output_deviation_rel"]["max"] for r in orbit_revs
                      if ko2s[r]["output_deviation_rel"]["max"] is not None]
            # zigzag interval count on the orbit: one zigzag per gauge sample over the orbit revisions
            zq = None
            if os.path.exists(orb_p) and len(orbit_revs) >= 2:
                try:
                    ob = np.load(orb_p); st = df.set_index("revision").step.to_dict()
                    ordr = sorted(orbit_revs, key=lambda r: st[r])
                    ng = min(ob[r].shape[0] for r in ordr)
                    counts = []
                    for j in range(ng):
                        zs = zigzag_summary([ob[r][j].astype(np.float64) for r in ordr],
                                            [st[r] for r in ordr], seed=0, verbose=False)
                        counts.append(len(zs["qualifying"]))
                    zq = dict(counts=counts, cv=cv_of(counts) if np.mean(counts) else 0.0, n_gauge=int(ng))
                    cvmax["zigzag_qualifying"] = zq["cv"]
                except Exception as e:
                    zq = dict(error=f"{type(e).__name__}: {e}")
            bad7 = {k: cvmax[k] for k in KO2_P7 if k in cvmax and not (cvmax[k] < ART_CV)}
            bad7r = {k: cvmax[k] for k in KO2_P7_REPAIRED if k in cvmax and not (cvmax[k] < ART_CV)}
            bad3 = {k: cvmax[k] for k in KO2_P3 if k in cvmax and not (cvmax[k] < ART_CV)}
            ver = dict(ran=True, n_gauge=int(ko2s[orbit_revs[0]]["n_gauge"]), revisions=orbit_revs,
                       device=ko2s[orbit_revs[0]].get("device"), torch_version=ko2s[orbit_revs[0]].get("torch_version"),
                       backend_dependence=("MEASURED AND UNRESOLVED (SPEC_O3 #O3-2): the EAP legibility quantities "
                                           "Q/treewidth/sparsity/community_sizes are density-thresholded graph "
                                           "statistics and their K-O-2 verdict has been observed to FLIP with the "
                                           "arithmetic backend on the same model and revisions (pythia-70m, step0 + "
                                           "step143000, 3 samples: Q CV 0.572 on CUDA, 1.4e-04 on CPU). Read this "
                                           "cell's legibility verdict as backend-conditional; the other P7 "
                                           "instruments and every P3 quantity sit at the fp32 floor on both."),
                       kind=GAUGE_KIND, orthogonal=GAUGE_ORTHOGONAL, art_cv=ART_CV,
                       fp32_floor_rel_max=(max(floors) if floors else None),
                       cv_max_over_revisions=cvmax, zigzag_orbit=zq,
                       artifacts_P7=bad7, artifacts_P3=bad3, evidence_only=KO2_EVIDENCE_ONLY,
                       repaired_instrument=dict(
                           quantities=KO2_P7_REPAIRED, artifacts=bad7r,
                           cv_max={k: cvmax.get(k) for k in KO2_P7_REPAIRED},
                           certifies=bool(not bad7r),
                           note="per-target-node EAP (SPEC_O3 #O3-2 / V AMENDMENTS #V7-1). Reported, NOT used to set "
                                "the P7 status: the pre-registered instrument is what O3.1 pre-registered, and "
                                "swapping the gating scalars after seeing a result would be a threshold change."),
                       backend="whiteboxtopo.symmetry.neox",
                       rule=(f"SPEC_O K-O-2: every certified scalar recomputed on {ko2s[orbit_revs[0]]['n_gauge']} "
                             f"random elements of G at every {a.orbit_every}-th revision plus the final; CV < "
                             f"{ART_CV} (or an identical set) -> certifiable, else 'failed (artifact)'. The fp32 "
                             f"output-deviation floor is the tolerance for 'identical'."))
    ev["k_o_2"] = ver

    # ---- the cells -----------------------------------------------------------------------------------------
    n_instr = len(P7_INSTRUMENTS)
    p7_val = dict(n_instruments=n_instr, n_detecting=len(names), detecting=names,
                  onsets_step=agree["onsets_step"], max_pairwise_delta_log10step=agree["max_pairwise_delta"],
                  concur=agree["concur"],
                  change_points={k: {kk: cps[v].get(kk) for kk in ("stat", "p", "onset_index", "onset_frac",
                                                                   "onset_step", "detected")}
                                 for k, v in P7_INSTRUMENTS.items() if isinstance(cps.get(v), dict)})
    cvm = ver.get("cv_max_over_revisions", {})
    p7_val["k_o_2"] = dict(ran=ver["ran"], artifacts=ver.get("artifacts_P7"),
                           cv_max={k: cvm.get(k) for k in KO2_P7},
                           fp32_floor_rel_max=ver.get("fp32_floor_rel_max"))
    p7_val["legibility_pre_registered"] = dict(
        instrument="mejiro_ardan.eap_graph (one gradient per layer) — the PRE-REGISTERED P7-d instrument; its K-O-2 "
                   "verdict is what sets this cell's status",
        trace="Q_excess at density 0.05", cv_max={k: cvm.get(k) for k in KO2_P7},
        n_distinct_edge_values=(int(df.n_distinct_edge_values.iloc[-1])
                                if "n_distinct_edge_values" in df.columns and len(df) else None),
        change_point=cps.get("Q_excess"))
    p7_val["legibility_repaired"] = dict(
        instrument="mejiro_ardan.eap_graph_per_node (one gradient per target node) — the SPEC_O3 #O3-2 repair, "
                   "reported alongside, never used to set the status",
        trace="Q_excess_pn at density 0.05", cv_max={k: cvm.get(k) for k in KO2_P7_REPAIRED},
        n_distinct_edge_values=(int(df.n_distinct_edge_values_pn.iloc[-1])
                                if "n_distinct_edge_values_pn" in df.columns and len(df) else None),
        certifies=ver.get("repaired_instrument", {}).get("certifies"),
        change_point=cps.get("Q_excess_pn"))
    p7_produced = any(isinstance(cps.get(v), dict) and "stat" in cps[v] for v in P7_INSTRUMENTS.values())
    p7_status = ko2_status(p7_produced, ver, ver.get("artifacts_P7"))
    p7_crit = (f"FINAL_SPEC §6.8 T3 = REPORTED, not pass/fail. Pre-registered positive (SPEC_O3 §6): >= 3 of the 4 "
               f"instruments detect a change point at p < {ALPHA} in BOTH models, their onsets concur within "
               f"{DELTA_LOG} decades of training step inside each model, and each concurring instrument agrees across "
               f"the two models. Status is the K-O-2 verdict (AMENDMENT #O3-1): certified iff the gauge orbit was run and no certified scalar moved by CV >= 1e-3; 'failed (artifact)' if one did; 'partial' if the orbit was not run.")

    bt_verified = b5_power_verified(model_id)
    kill_rows = df[df.b2_kill_pass == 1] if "b2_kill_pass" in df.columns else df.iloc[0:0]
    p3_val = dict(
        b2=dict(kill_pass_any_revision=bool(len(kill_rows)),
                revisions_passing_kill=[str(r) for r in kill_rows.revision] if len(kill_rows) else [],
                cos_mf_disjoint_halves_max=float(df.cos_mf_disjoint_halves.max()) if len(df) else None,
                cos_random_split_absmean_min=float(df.cos_random_split_absmean.min()) if len(df) else None,
                mlp_hidden_and_per_head_coordinates="computed and audited on the gauge orbit (AMENDMENT #O3-1); see "
                                                    "coordinate_audits below",
                change_point_cos_to_final=cps.get("cos_to_final"), change_point_dbar=cps.get("dbar")),
        b3=dict(n_per_group=int(df.biastop_n.iloc[-1]) if len(df) else None, power_verified=bool(bt_verified),
                n_source=("B5 hpcc/results/b5/B5_requirements.json" if bt_verified else
                          f"fallback n = {N_BIAS_FALLBACK} (B5 has no power-0.8 requirement for this family; GPT-2's "
                          f"B3 never reached power 0.8 at any feasible n — max 0.60 at stretch delta 0.25, 0.25 on the "
                          f"real M/F split). A non-rejection here is UNINFORMATIVE, not evidence of no bias."),
                n_revisions_rejecting=int((df.biastop_p < ALPHA).sum()) if "biastop_p" in df.columns else None,
                min_p=float(df.biastop_p.min()) if "biastop_p" in df.columns else None,
                change_point_biastop=cps.get("biastop_stat")),
        onset_answer_step=(cps.get("cos_to_final", {}).get("onset_step") if cps.get("cos_to_final", {}).get("detected")
                           else cps.get("dbar", {}).get("onset_step") if cps.get("dbar", {}).get("detected") else None))
    p3_val["b2"]["coordinate_audits"] = dict(
        ran=ver["ran"], note=("g in MLP-hidden and per-head output coordinates — the objects B2's audit actually "
                              "stresses — are now computable and ARE audited (AMENDMENT #O3-1); aligned with the "
                              "exact aligner, un-permute FIRST then un-rotate at the base head index "
                              "(B AMENDMENTS #4's ordering trap)"),
        cv_max={k: ver.get("cv_max_over_revisions", {}).get(k)
                for k in ("g_resid_cos", "g_mlp_norm", "g_mlp_cos_aligned", "g_heads_cos_aligned")},
        evidence_only_g_heads_per_head_norm=ver.get("cv_max_over_revisions", {}).get("g_heads_per_head_norm"))
    p3_val["k_o_2"] = dict(ran=ver["ran"], artifacts=ver.get("artifacts_P3"),
                           cv_max={k: ver.get("cv_max_over_revisions", {}).get(k) for k in KO2_P3},
                           fp32_floor_rel_max=ver.get("fp32_floor_rel_max"))
    p3_status = ko2_status("b2_kill_pass" in df.columns and len(df) > 0, ver, ver.get("artifacts_P3"))
    p3_crit = (f"FINAL_SPEC §6.4 T3 = REPORTED. Pre-registered positive (SPEC_O3 §6), all three: (i) the B2 kill test "
               f"passes at >= 1 revision (disjoint-halves cosine > 0.9 AND random-split |cosine| < 0.5); (ii) a bias "
               f"trace (cos_to_final or dbar) has a change point at p < {ALPHA}, whose onset_step is the answer to "
               f"'when bias entered'; (iii) BiasTop rejects at alpha = {ALPHA} at an n whose power >= 0.8 is "
               f"established by B5. Without (iii) the reported RESULT is not a positive whatever (i) and (ii) do — independently of the cell's K-O-2 status, which is about gauge invariance, not about power.")

    card = Scorecard(model=model_id, behavior=dict(name="next-token prediction on the frozen OpenWebText-10k probe",
                                                   kind="lm", probe_tokens=int(a.n_seq * (SEQ_TOK - 1)),
                                                   revisions=int(len(df))))
    for p, why in NOT_REACHED.items():
        card.cells[p] = CellResult(p, "not_reached", provenance=f"not reached here: {why}")
    card.cells["P7"] = CellResult("P7", p7_status, value=p7_val,
                                  evidence={"criterion": p7_crit, "agreement": agree,
                                            "instruments": list(P7_INSTRUMENTS)},
                                  null=ev["nulls"], provenance=GAUGE_PROV)
    card.cells["P3"] = CellResult("P3", p3_status, value=p3_val,
                                  evidence={"criterion": p3_crit,
                                            "corpus": f"blib.build_corpus_unique: {len(df)} revisions x "
                                                      f"{int(df.n_male.iloc[-1]) if len(df) else 0} M + "
                                                      f"{int(df.n_female.iloc[-1]) if len(df) else 0} F distinct "
                                                      f"sentences (B AMENDMENTS #2)"},
                                  null=ev["nulls"], provenance=GAUGE_PROV)
    card.gamma_cov = None
    card.reconstruction = None
    card.errors = ([] if ver.get("ran") else
                   [f"K-O-2 not run in this pass ({ver.get('reason', '')}) — P3/P7 stay 'partial' "
                    f"(gauge-fixed, not gauge-certified). Re-run with --n-gauge > 0 (AMENDMENT #O3-1)."])
    for p, bad in (("P7", ver.get("artifacts_P7")), ("P3", ver.get("artifacts_P3"))):
        if bad:
            card.errors.append(f"{p} failed K-O-2 (artifact): {bad} — CV >= {ART_CV} on the gauge orbit")
    card.provenance = dict(spec="experiments/O_opener/SPEC_O3.md", runner="experiments/O_opener/hishi_akebono.py",
                           trajectory=os.path.relpath(traj_p, ROOT), clouds=os.path.relpath(cl_p, ROOT),
                           evidence=os.path.relpath(os.path.join(a.out, f"O31_{tag}_cells.json"), ROOT))
    card.config = dict(n_seq=a.n_seq, n_cloud=a.n_cloud, n_pc=N_PC, knn_k=KNN_K, n_eigs=N_EIGS,
                       small_threshold=SMALL_THR, simplex_budget=SIMPLEX_BUDGET, zz_max_ck=ZZ_MAX_CK,
                       alpha=ALPHA, n_perm=a.n_perm, label_null=a.label_null, biastop_nperm=a.biastop_nperm,
                       densities=list(DENS), density_primary=DENS_PRIMARY, delta_log_step=DELTA_LOG,
                       n_gauge=a.n_gauge, orbit_every=a.orbit_every, art_cv=ART_CV, gauge_kind=GAUGE_KIND,
                       gauge_orthogonal=GAUGE_ORTHOGONAL, gauge_backend="whiteboxtopo.symmetry.neox")

    df.to_csv(traj_p, index=False)
    json.dump(ev, open(os.path.join(a.out, f"O31_{tag}_cells.json"), "w"), indent=1, default=str)
    json.dump(card.to_dict(), open(os.path.join(a.out, f"scorecard_{tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out, f"scorecard_{tag}.md"), "w").write(card.to_markdown() + "\n")
    print(card.to_markdown(), flush=True)
    return card.to_dict()


# ========================================================================================================= aggregate
def aggregate(a):
    """The two-model comparison of SPEC_O3 §6: per-instrument onset agreement and trace agreement."""
    cards = {}
    for p in sorted(glob.glob(os.path.join(a.out, "scorecard_*.json"))):
        d = json.load(open(p)); cards[d["model"]] = d
    if len(cards) < 2:
        print(f"aggregate: {len(cards)} scorecard(s) in {a.out}; need 2 — writing what exists", flush=True)
    evs = {}
    for m in cards:
        p = os.path.join(a.out, f"O31_{tag_of(m)}_cells.json")
        if os.path.exists(p):
            evs[m] = json.load(open(p))
    models = sorted(cards)
    comp = dict(models=models, spec="experiments/O_opener/SPEC_O3.md", tolerance_log10step=DELTA_LOG,
                per_instrument={}, trace_agreement={}, summary={})
    if len(models) == 2:
        m1, m2 = models
        d1 = pd.read_csv(os.path.join(a.out, f"O31_{tag_of(m1)}_trajectory.csv")).sort_values("step")
        d2 = pd.read_csv(os.path.join(a.out, f"O31_{tag_of(m2)}_trajectory.csv")).sort_values("step")
        shared = sorted(set(d1.revision) & set(d2.revision), key=lambda r: int(r[4:]))
        d1 = d1[d1.revision.isin(shared)].sort_values("step").reset_index(drop=True)
        d2 = d2[d2.revision.isin(shared)].sort_values("step").reset_index(drop=True)
        comp["n_shared_revisions"] = len(shared)
        n_agree = n_both = 0
        for label, trace in P7_INSTRUMENTS.items():
            c1 = evs.get(m1, {}).get("change_points", {}).get(trace, {})
            c2 = evs.get(m2, {}).get("change_points", {}).get(trace, {})
            row = dict(trace=trace,
                       m1=dict(model=m1, p=c1.get("p"), detected=c1.get("detected"), onset_step=c1.get("onset_step")),
                       m2=dict(model=m2, p=c2.get("p"), detected=c2.get("detected"), onset_step=c2.get("onset_step")))
            if c1.get("detected") and c2.get("detected"):
                n_both += 1
                dl = abs(log_step(c1["onset_step"]) - log_step(c2["onset_step"]))
                row["delta_log10step"] = round(float(dl), 4); row["agree"] = bool(dl <= DELTA_LOG)
                n_agree += int(dl <= DELTA_LOG)
            comp["per_instrument"][label] = row
        for trace in TRACES + ["Q", "treewidth_ub", "sparsity", "np_H1"]:
            if trace not in d1.columns or trace not in d2.columns:
                continue
            x, y = d1[trace].to_numpy(float), d2[trace].to_numpy(float)
            if len(x) < 6 or not np.isfinite(x).all() or not np.isfinite(y).all() or x.std() == 0 or y.std() == 0:
                comp["trace_agreement"][trace] = dict(note="trace too short, constant or incomplete in one model")
                continue
            rho = float(spearmanr(x, y).statistic)
            rng = np.random.default_rng(0)
            null = np.array([abs(float(spearmanr(x, rng.permutation(y)).statistic)) for _ in range(a.n_perm // 4)])
            comp["trace_agreement"][trace] = dict(spearman=rho, n=int(len(x)),
                                                  p=float((1 + np.sum(null >= abs(rho))) / (len(null) + 1)),
                                                  n_perm=int(len(null)), null="checkpoint-order shuffle")
        comp["summary"] = dict(n_p7_instruments=len(P7_INSTRUMENTS), n_detecting_in_both=n_both, n_agreeing=n_agree,
                               p7_positive=bool(n_both >= 3 and n_agree == n_both and
                                                all(evs.get(m, {}).get("p7_agreement", {}).get("concur")
                                                    for m in models)),
                               note=("P7 positive requires >= 3 of 4 instruments detecting in BOTH models, concurring "
                                     "within each model, and agreeing across models (SPEC_O3 §6). Nothing here is "
                                     "'certified': K-O-2 is unrunnable on gpt-neox."))
    for m in models:
        comp.setdefault("scorecards", {})[m] = {p: cards[m]["cells"][p]["status"] for p in cards[m]["cells"]}
    json.dump(comp, open(os.path.join(a.out, "O31_compare.json"), "w"), indent=1, default=str)
    flat = []
    for label, row in comp["per_instrument"].items():
        flat.append(dict(instrument=label, trace=row["trace"],
                         **{f"{k}_{kk}": row[k][kk] for k in ("m1", "m2") for kk in ("model", "p", "detected",
                                                                                     "onset_step")},
                         delta_log10step=row.get("delta_log10step"), agree=row.get("agree")))
    pd.DataFrame(flat).to_csv(os.path.join(a.out, "O31_compare.csv"), index=False)
    print(json.dumps(comp.get("summary", {}), indent=1), flush=True)
    return comp


# ============================================================================================================== main
def main():
    ap = argparse.ArgumentParser(description="O3.1 Hishi Akebono — T3 Pythia P7/P3 (SPEC_O3.md)")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revisions", nargs="*", default=None, help="default: the 154 V7 revisions")
    ap.add_argument("--out", default="hpcc/results/o31")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 revisions, tiny probe, reduced nulls")
    ap.add_argument("--aggregate", action="store_true", help="compare the two models from their scorecards")
    ap.add_argument("--finalise", action="store_true", help="re-run the trajectory instruments only (no GPU)")
    ap.add_argument("--purge", action="store_true", help="delete each revision's HF snapshot after use (needed on a "
                                                         "small scratch; never alongside V7 on the same HF_HOME)")
    ap.add_argument("--n-seq", type=int, default=N_SEQ)
    ap.add_argument("--n-cloud", type=int, default=N_CLOUD)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--label-null", type=int, default=N_PERM_LABEL)
    ap.add_argument("--biastop-nperm", type=int, default=BIASTOP_NPERM)
    ap.add_argument("--bias-n", type=int, default=0, help="n per group for BiasTop (0 = B5's requirement, else the "
                                                          "fallback)")
    ap.add_argument("--bias-n-sent", type=int, default=0, help="cap the probe corpus per group (smoke only)")
    ap.add_argument("--n-gauge", type=int, default=N_GAUGE, help="K-O-2 gauge samples per orbit revision (0 = skip; "
                                                                "the cells then stay 'partial')")
    ap.add_argument("--orbit-every", type=int, default=ORBIT_EVERY, help="K-O-2 orbit = every Nth revision + the final")
    ap.add_argument("--zz-null", type=int, default=0, help="zigzag cloud-order permutations (0 = not run)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not os.path.isabs(a.out):
        a.out = os.path.join(ROOT, a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        a.revisions = a.revisions or SMOKE_REVISIONS
        a.n_seq = min(a.n_seq, 4); a.n_cloud = min(a.n_cloud, 120); a.n_perm = min(a.n_perm, 200)
        a.label_null = min(a.label_null, 50); a.biastop_nperm = min(a.biastop_nperm, 25)
        a.bias_n = a.bias_n or 40; a.bias_n_sent = a.bias_n_sent or 120
        a.n_gauge = min(a.n_gauge, 3); a.orbit_every = min(a.orbit_every, 2)
    if a.aggregate:
        aggregate(a); return
    revs = a.revisions or REVISIONS
    if a.finalise:
        tag = tag_of(a.model)
        finalise(a, a.model, os.path.join(a.out, f"O31_{tag}_trajectory.csv"),
                 os.path.join(a.out, f"clouds_{tag}.npz")); return
    t0 = time.time()
    run_model(a, a.model, revs)
    print(f"done {a.model}: {len(revs)} revisions requested, {round(time.time() - t0, 1)}s wall", flush=True)


if __name__ == "__main__":
    main()
