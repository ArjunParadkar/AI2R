#!/usr/bin/env python3
"""O1.4 Fine Motion — P4 ("what it does not know") and P5 ("where it will fail") on the T1 specimens.

  python fine_motion.py [--specimens grok_clean_s0 ring_s0 ...] [--cells P4 P5] [--n-gauge 20] [--force]

Pre-registration: SPEC_O.md §O1.4 + kill test K-O-2; FINAL_SPEC §2.10, §2.11, §6.5, §6.6.
Realisation of the inputs/estimators on T1: AMENDMENTS.md #O1.4 (a)-(h), written before any number here was computed, plus
the post-hoc addenda #O1.4 (i)-(iv) (diagnostics that never change a criterion: the extended radius grid, the P4 ambient
nearest-neighbour diagnostics, and the ring's standardised-coordinate / transported-query gauge fixing).

P4 (grok): diffusion-embedding OOD distance d_M (Nystrom) on the in-distribution MLP activation cloud at the '='
position (training pairs), scored on held-out pairs and on six OOD conditions (corrupt_eq, swap_eq, dup_eq, oov_emb,
shortcut_tok [planted only], adversarial). Pass: AUC > 0.9 separating held-out from OOD, beating max-softmax
confidence AND the logit margin; null: spectral.random_projection_null must not reach the same AUC.
P4 (ring): in-distribution = hidden states of held-out sequences at the training velocity sigma; OOD = states driven at
3 sigma and states kicked off the attractor; baseline = |‖readout‖ − 1|.

P5 (grok): rho(x) = ‖Q_h(x)‖ · Lip(x) (curvature of the MLP cloud at h(x) × spectral norm of the autograd Jacobian of
the last-position logits w.r.t. the embedded input — AMENDMENTS #O1.4c) vs the measured failure radius (Gaussian
perturbation of the embedded input at r ∈ {0.01,…,0.5} relative to the global std of the embedded inputs; failure =
argmax flip on ≥ 5 of 8 draws — AMENDMENTS #O1.4d). Pass: Spearman < −0.5 at p < 0.01, beating the softmax-margin
baseline; null: shuffled rho. The pre-registered grid has no dynamic range on a converged grokked specimen, so a second,
pre-registered measurement of the same quantity runs beside it: **P5 v2**, the per-input failure radius by 12-step
bisection on [0, 4 sigma] under the same majority-of-8 rule (AMENDMENTS #O1.4 addenda (v)). Thresholds, baseline, null
and rho are identical; v2 appears as `evidence.v2` and as `_v2` CSV rows. The cell status is always the v1 verdict.
P5 (ring): rho = ‖Q_h‖ × ‖J_step(h)‖ in the per-unit-standardised coordinates; failure = angular error > 0.3 rad vs
the unperturbed model after 50 further steps. Ring specimens are run in float64 (see ring_states).

K-O-2: every certified quantity (AUCs, both Spearmans) is recomputed on the gauge orbit — O1.4 ships at n_gauge = 20
(AMENDMENTS #O1.4f), stated in every cell's provenance and every CSV row;
clouds are standardised per unit by the in-distribution scales (the recorded gauge fixing, AMENDMENTS #O1.4h); CV > 1e-3
downgrades the cell to "failed (artifact)".

Outputs: results/fine_motion_P4.csv, results/fine_motion_P5.csv (one row per specimen × condition × scorer /
predictor) and results/open/fine_motion_<specimen>.json. Resumable: a specimen whose JSON exists is skipped
unless --force.
"""
from __future__ import annotations
import argparse, copy, csv, json, os, sys, time, traceback
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

torch.set_num_threads(4)                       # the GPU box is shared; keep BLAS off the other jobs' cores

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import olib  # noqa: E402
from olib import RES  # noqa: E402
from sweep_tosho import P, EQ, SHORT0, N_SHORT, ring_batch, RingRNN  # noqa: E402
from whiteboxtopo.spectral import out_of_manifold, random_projection_null  # noqa: E402
from whiteboxtopo.geometry import curvature  # noqa: E402

OUT = os.path.join(RES, "open"); os.makedirs(OUT, exist_ok=True)
P4_CSV = os.path.join(RES, "fine_motion_P4.csv")
P5_CSV = os.path.join(RES, "fine_motion_P5.csv")
P4_FIELDS = ["specimen", "kind", "condition", "scorer", "auc", "n_in", "n_ood", "beats_baselines", "beats_null",
             "status", "cv_auc", "n_gauge", "wall"]
P5_FIELDS = ["specimen", "kind", "predictor", "spearman", "p_value", "n", "frac_inf", "beats_baseline",
             "null_p", "status", "cv_spearman", "n_gauge", "wall"]

RADII = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)     # SPEC_O O1.4, relative (AMENDMENTS #O1.4d) — the pre-registered grid
RADII_EXT = RADII + (1.0, 2.0, 5.0, 10.0)      # DIAGNOSTIC ONLY when the grid above has no dynamic range (AMENDMENTS #O1.4i)
R_MAX_V2 = 4.0                                 # P5 v2 bisection bracket, relative (AMENDMENTS #O1.4 addenda (v))
BISECT_STEPS = 12
N_DIRS = 8
MAJORITY = 5                                   # >= 5 of 8 draws flip  (AMENDMENTS #O1.4d)
CV_MAX = 1e-3                                  # K-O-2
AUC_MIN = 0.9                                  # FINAL_SPEC 6.5
SPEARMAN_MAX = -0.5                            # FINAL_SPEC 6.6
P_MAX = 0.01
RING_FAIL_RAD = 0.3                            # rad, SPEC_O O1.4 ring P5
RING_ROLL = 50
N_CLOUD = 1200
N_QUERY = 256
N_P5 = 256
K_CURV = 30                                    # AMENDMENTS #O1.4h
N_BG = 512                                     # background points padding the curvature cloud


# --------------------------------------------------------------------------------------------- cell plumbing
def cell(status, value=None, criterion="", provenance="", **evidence):
    return dict(status=status, value=value, criterion=criterion, provenance=provenance, evidence=evidence)


def guarded(fn):
    def run(*a, **k):
        t = time.time()
        try:
            c = fn(*a, **k)
        except Exception as e:
            c = cell("not_reached", provenance=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1200:])
        c["wall"] = round(time.time() - t, 1); return c
    return run


def auc_of(score_in, score_ood):
    """AUC with label 1 = OOD, higher score = more OOD. nan if either side is empty."""
    si, so = np.asarray(score_in, float).ravel(), np.asarray(score_ood, float).ravel()
    if si.size == 0 or so.size == 0: return float("nan")
    y = np.r_[np.zeros(si.size), np.ones(so.size)]
    s = np.r_[si, so]
    s = np.where(np.isfinite(s), s, np.nanmax(s[np.isfinite(s)]) * 10 if np.isfinite(s).any() else 0.0)
    return float(roc_auc_score(y, s))


def std_scales(cloud, eps=0.0):
    """topology.scaling.standardized_cloud convention: per-unit std, zero-scale units left alone (divide by 1)."""
    s = np.asarray(cloud, float).std(0)
    return np.where(s <= eps, 1.0, s)


# --------------------------------------------------------------------------------------------- toy-transformer forward
def embed(m, tokens):
    """Embedded input x = wte[tokens] + wpe (B, T, d) — the object P4's OOD conditions and P5's perturbations live in.
    No element of the gauge group touches wte/wpe, so x is gauge-invariant."""
    T = tokens.shape[1]
    return m.wte[tokens] + m.wpe[:T][None]


def forward_embed(m, x, want_mlp=False):
    """Logits at the last position from an already-embedded input x (B, T, d). Mirrors ToyTransformer.forward."""
    T = x.shape[1]; mlp_last = None
    for l in range(m.L):
        h = m._ln(x, m.ln1_g[l], m.ln1_b[l])
        q = torch.einsum("btd,dhk->bthk", h, m.Wq[l]) + m.bq[l]
        k = torch.einsum("btd,dhk->bthk", h, m.Wk[l]) + m.bk[l]
        v = torch.einsum("btd,dhk->bthk", h, m.Wv[l]) + m.bv[l]
        att = (torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(m.dh)).masked_fill(~m.mask[:T, :T], float("-inf")).softmax(-1)
        z = torch.einsum("bhts,bshk->bthk", att, v)
        x = x + torch.einsum("bthk,hkd->bthd", z, m.Wo[l]).sum(2) + m.bo[l]
        h2 = m._ln(x, m.ln2_g[l], m.ln2_b[l])
        a = torch.relu(h2 @ m.W1[l] + m.b1[l])
        mlp_last = a[:, -1]
        x = x + a @ m.W2[l] + m.b2[l]
    logits = (m._ln(x, m.lnf_g, m.lnf_b) @ m.Wu + m.bu)[:, -1]
    return (logits, mlp_last) if want_mlp else logits


@torch.no_grad()
def acts_logits(m, x, batch=1024):
    """(mlp cloud at '=', logits) for embedded inputs x, as numpy."""
    A, L = [], []
    for i in range(0, len(x), batch):
        lg, a = forward_embed(m, x[i:i + batch], want_mlp=True)
        A.append(a.detach().float().cpu().numpy()); L.append(lg.detach().float().cpu().numpy())
    return np.concatenate(A).astype(np.float64), np.concatenate(L).astype(np.float64)


def logit_baselines(L):
    """(max-softmax OOD score, logit-margin OOD score): both oriented so that larger = more OOD / more fragile."""
    Z = L - L.max(1, keepdims=True)
    pmax = np.exp(Z).max(1) / np.exp(Z).sum(1)
    S = np.sort(L, 1)
    margin = S[:, -1] - S[:, -2]
    return -pmax, -margin, margin



def nn_diagnostics(C, QA, inn, sl, ood, pooled):
    """Post-hoc diagnostics of a P4 outcome (never a criterion): the ambient nearest-reference distance of each query
    group, the cloud's own nearest-neighbour distance, and the AUC of that raw ambient distance. They separate
    "the diffusion embedding lost the signal" from "the activation cloud never placed these inputs further out"."""
    from scipy.spatial.distance import cdist
    D = cdist(QA, C)
    dref = cdist(C, C); np.fill_diagonal(dref, np.inf)
    nn = D.min(1)
    t = {c: auc_of(nn[inn], nn[sl[c]]) for c in ood}; t["pooled"] = auc_of(nn[inn], nn[pooled])
    return {"auc_ambient_nn": t,
            "nn_dist_median": {c: float(np.median(nn[sl[c]])) for c in list(sl)},
            "cloud_own_nn_dist_median": float(np.median(dref.min(1))),
            "cloud_norm_median": float(np.median(np.linalg.norm(C, axis=1)))}


# --------------------------------------------------------------------------------------------- P4 on the grok specimens
def grok_p4_inputs(m, data, seed=0):
    """Embedded query inputs per condition (AMENDMENTS #O1.4a/#6b) plus the in-distribution cloud tokens."""
    rng = np.random.default_rng(seed)
    tr, te = data["tr"], data["te"]
    i_cloud = tr[rng.permutation(len(tr))[:N_CLOUD]]
    i_q = te[rng.permutation(len(te))[:N_QUERY]]
    toks_cloud = olib.grok_tokens(data, i_cloud)
    toks_held = olib.grok_tokens(data, i_q)
    T = toks_held.shape[1]
    eq_pos = T - 1
    conds = {}
    conds["heldout"] = embed(m, toks_held).detach()
    t = toks_held.clone(); t[:, eq_pos] = torch.as_tensor(rng.integers(0, P, size=len(t)), device=t.device)
    conds["corrupt_eq"] = embed(m, t).detach()                                     # random number token in place of '='
    t = toks_held.clone()
    order = [1, eq_pos, 0] + [j for j in range(T) if j not in (0, 1, eq_pos)]      # [b, =, a] (+ s for planted)
    conds["swap_eq"] = embed(m, t[:, order]).detach()
    t = toks_held.clone(); t[:, 1] = EQ
    conds["dup_eq"] = embed(m, t).detach()                                         # duplicated '='
    x = embed(m, toks_held).detach().clone()
    nrm = float(m.wte.detach().norm(dim=1).mean())
    g = torch.as_tensor(rng.standard_normal((len(x), m.d)), dtype=x.dtype, device=x.device)
    g = g / g.norm(dim=1, keepdim=True) * nrm
    x[:, 0] = g + m.wpe[0][None]                                                   # OOV direction of matched norm
    conds["oov_emb"] = x
    if m.vocab > P + 1:                                                            # planted only (AMENDMENTS #O1.4b)
        t = toks_held.clone(); t[:, 0] = torch.as_tensor(rng.integers(SHORT0, SHORT0 + N_SHORT, size=len(t)), device=t.device)
        conds["shortcut_tok"] = embed(m, t).detach()
    return toks_cloud, conds, i_cloud, i_q


def adversarial_embed(m, x0, radii=RADII, steps=10, seed=0):
    """Gradient-ascent perturbation in embedding space that flips the argmax, at the smallest grid radius that works
    (AMENDMENTS #O1.4a-vi). Returns (x_adv, mask_flipped, radius_used)."""
    sigma = float(x0.std())
    n, T, d = x0.shape
    with torch.no_grad():
        base = forward_embed(m, x0).argmax(1)
    adv = x0.clone(); used = np.full(n, np.nan); todo = torch.ones(n, dtype=torch.bool, device=x0.device)
    for r in radii:
        if not bool(todo.any()): break
        eps = r * sigma * np.sqrt(T * d)
        xa = x0.clone()
        for _ in range(steps):
            xa = xa.detach().requires_grad_(True)
            lg = forward_embed(m, xa)
            loss = torch.nn.functional.cross_entropy(lg, base)
            gr, = torch.autograd.grad(loss, xa)
            gf = gr.reshape(n, -1); gf = gf / (gf.norm(dim=1, keepdim=True) + 1e-12)
            xa = xa.detach() + (0.25 * eps) * gf.reshape(n, T, d)
            dlt = (xa - x0).reshape(n, -1); nd = dlt.norm(dim=1, keepdim=True)
            dlt = dlt * torch.clamp(eps / (nd + 1e-12), max=1.0)
            xa = x0 + dlt.reshape(n, T, d)
        with torch.no_grad():
            flip = forward_embed(m, xa).argmax(1) != base
        hit = todo & flip
        adv[hit] = xa[hit].detach(); used[hit.cpu().numpy()] = r; todo = todo & ~flip
    ok = ~np.isnan(used)
    return adv, ok, used


@guarded
def grok_P4(m, data, name, n_gauge, seed=0):
    toks_cloud, conds, i_cloud, i_q = grok_p4_inputs(m, data, seed)
    x_cloud = embed(m, toks_cloud).detach()
    adv, adv_ok, adv_r = adversarial_embed(m, conds["heldout"], seed=seed)
    if adv_ok.sum() >= 10:
        conds["adversarial"] = adv[torch.as_tensor(adv_ok, device=adv.device)]
    order = [c for c in conds]
    Q = torch.cat([conds[c] for c in order])
    sizes = [len(conds[c]) for c in order]

    def scores(model):
        cloud, _ = acts_logits(model, x_cloud)
        qa, ql = acts_logits(model, Q)
        sc = std_scales(cloud)
        res = out_of_manifold(cloud / sc[None, :], qa / sc[None, :], k=15, t=1, n_eigs=10, seed=seed)
        nul = random_projection_null(cloud / sc[None, :], qa / sc[None, :], d=10, seed=seed)
        conf, negmargin, _ = logit_baselines(ql)
        return res, nul, conf, negmargin, cloud

    res, nul, conf, negmargin, cloud = scores(m)
    cut = np.cumsum([0] + sizes)
    sl = {c: slice(cut[i], cut[i + 1]) for i, c in enumerate(order)}
    ood = [c for c in order if c != "heldout"]
    pooled = np.concatenate([np.arange(cut[i], cut[i + 1]) for i, c in enumerate(order) if c != "heldout"]) if ood else np.array([], int)
    inn = np.arange(cut[0], cut[1])

    def table(scr):
        t = {c: auc_of(scr[inn], scr[sl[c]]) for c in ood}
        t["pooled"] = auc_of(scr[inn], scr[pooled])
        return t

    A = table(res.d_M); Acon = table(conf); Amar = table(negmargin); Anul = table(nul.d_M)
    beats = {c: bool(A[c] > max(Acon[c], Amar[c])) for c in A}
    inverted = {c: bool(A[c] < 0.5) for c in A}
    beats_null = {c: bool(A[c] > Anul[c]) for c in A}
    ok = A["pooled"] > AUC_MIN and beats["pooled"] and beats_null["pooled"]

    # ---- K-O-2 gauge orbit on the certified quantity (the AUC vector)
    def auc_vec(model):
        r, _, _, _, _ = scores(model)
        t = table(r.d_M); return np.array([t[c] for c in ["pooled"] + ood])
    orbit = olib.on_orbit(m, auc_vec, n=n_gauge, seed=41)
    cvv = olib.cv(np.asarray(orbit))
    status = "certified" if (ok and cvv < CV_MAX) else ("failed (artifact)" if cvv >= CV_MAX else "failed")
    diag = {}
    if status != "certified":                                   # diagnose the negative; none of this is a criterion
        sc0 = std_scales(cloud); qa0, _ = acts_logits(m, Q)
        diag = nn_diagnostics(cloud / sc0[None, :], qa0 / sc0[None, :], inn, sl, ood, pooled)
        try:                                                    # is it the 1200-point subsample? re-score on every training pair
            xf = embed(m, olib.grok_tokens(data, data["tr"])).detach()
            cf, _ = acts_logits(m, xf); scf = std_scales(cf)
            rf = out_of_manifold(cf / scf[None, :], qa0 / scf[None, :], k=15, t=1, n_eigs=10, seed=seed)
            tf = table(rf.d_M); diag["auc_full_training_cloud"] = tf; diag["n_ref_full"] = int(rf.n_ref)
        except Exception as e:
            diag["auc_full_training_cloud"] = f"not_reached: {type(e).__name__}: {e}"
    return cell(status, value=dict(auc=A["pooled"], auc_maxsoftmax=Acon["pooled"], auc_margin=Amar["pooled"], auc_null=Anul["pooled"]),
                criterion=f"AUC(d_M, held-out vs pooled OOD) > {AUC_MIN} AND > max(max-softmax, logit-margin) AUC AND > random-projection-null AUC; K-O-2 CV < {CV_MAX}",
                provenance=("spectral.out_of_manifold (Nystrom, k=15, t=1, n_eigs=10) on the MLP activation cloud at the '=' position; "
                            f"in-distribution cloud = {len(i_cloud)} training pairs, queries = {N_QUERY} held-out pairs per condition; "
                            "cloud standardised per unit by the in-distribution std (gauge fixing, AMENDMENTS #O1.4h); "
                            f"OOD conditions per AMENDMENTS #O1.4a/#6b; null = spectral.random_projection_null(d=10); gauge orbit n={n_gauge} (AMENDMENTS #O1.4f)"),
                auc=A, auc_maxsoftmax=Acon, auc_margin=Amar, auc_null=Anul, beats_baselines=beats, beats_null=beats_null,
                n_in=int(len(inn)), n_ood={c: int(sizes[order.index(c)]) for c in ood}, cv_auc=cvv, n_gauge=n_gauge,
                threshold=float(res.threshold), frac_flagged_heldout=float(res.flags[inn].mean()),
                frac_flagged={c: float(res.flags[sl[c]].mean()) for c in ood},
                adversarial_flip_rate=float(adv_ok.mean()), adversarial_radius_median=float(np.nanmedian(adv_r)) if adv_ok.any() else None,
                conditions_not_reached=({"shortcut_tok": "vocab is p+1: no out-of-vocabulary token id exists (AMENDMENTS #O1.4b)"}
                                        if m.vocab <= P + 1 else {}) | ({} if "adversarial" in conds else {"adversarial": "fewer than 10 inputs flipped inside the radius grid"}),
                cloud_dim=int(cloud.shape[1]), dead_units=int((cloud.std(0) <= 0).sum()), inverted=inverted, diagnostics=diag)


# --------------------------------------------------------------------------------------------- P5 on the grok specimens
def jacobian_lip(m, x, batch=256):
    """‖J‖_2 of the last-position logits w.r.t. the embedded input (AMENDMENTS #O1.4c): n_out batched backward passes.
    ‖A‖_2 = sqrt(lambda_max(A A^T)) with A (n_out, T*d) — the Gram route, since n_out << T*d is not required but the
    Gram is the smaller of the two and the eigenproblem is 113x113."""
    lips = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].detach().clone().requires_grad_(True)
        lg = forward_embed(m, xb)
        rows = []
        for c in range(lg.shape[1]):
            g, = torch.autograd.grad(lg[:, c].sum(), xb, retain_graph=(c < lg.shape[1] - 1))
            rows.append(g.reshape(len(xb), -1))
        A = torch.stack(rows, 1).float()                             # (b, n_out, T*d)
        G = torch.bmm(A, A.transpose(1, 2))                          # (b, n_out, n_out)
        lips.append(torch.sqrt(torch.clamp(torch.linalg.eigvalsh(G)[:, -1], min=0.0)).double().cpu().numpy())
    return np.concatenate(lips)


def grid_radius(flip, n, radii=RADII):
    """v1 (AMENDMENTS #O1.4d): smallest radius on the pre-registered grid at which the majority rule fires."""
    out = np.full(n, np.inf); todo = np.ones(n, bool)
    for r in radii:
        if not todo.any(): break
        hit = todo & flip(np.full(n, float(r)))
        out[hit] = r; todo &= ~hit
    return out


def bisect_radius(flip, n, r_max=R_MAX_V2, steps=BISECT_STEPS):
    """v2 (AMENDMENTS #O1.4 addenda (v)): per-input failure radius by `steps`-step bisection on [0, r_max] under the
    same majority-of-8 rule. Inputs that do not fail at r_max get inf. Bisection assumes the flip indicator is monotone
    in r; it is not guaranteed to be, so the return is the upper end of the final bracket — "a radius at which the rule
    fires, with no smaller firing radius found along the bisection path" — not a certified minimum."""
    rad = np.full(n, np.inf)
    top = flip(np.full(n, float(r_max)))
    rad[top] = r_max
    lo = np.zeros(n); hi = np.where(top, float(r_max), np.inf); live = top.copy()
    for _ in range(steps):
        if not live.any(): break
        mid = np.where(live, 0.5 * (lo + hi), float(r_max))
        f = flip(mid)
        up = live & f; lo_m = live & ~f
        hi[up] = mid[up]; rad[up] = mid[up]
        lo[lo_m] = mid[lo_m]
    return rad


def grok_perturbation(m, x, n_dirs=N_DIRS, majority=MAJORITY, seed=0):
    """(flip, sigma) for the embedded inputs x. flip(r) with r a per-input relative radius -> bool (n,), True where
    >= `majority` of `n_dirs` Gaussian draws flip the argmax (AMENDMENTS #O1.4d). Nested design: the directions are
    drawn once, normalised to the unit sphere and reused at every radius; the perturbation norm is
    r * sigma * sqrt(T*d) with sigma the global std of the embedded inputs, so each draw has per-coordinate scale
    r*sigma. The embedded input is gauge-invariant (no element of G touches wte/wpe), so flip is too."""
    n, T, d = x.shape
    sigma = float(x.std())
    g = torch.as_tensor(np.random.default_rng(seed).standard_normal((n, n_dirs, T * d)), dtype=x.dtype, device=x.device)
    g = (g / g.norm(dim=2, keepdim=True)).reshape(n, n_dirs, T, d)
    with torch.no_grad():
        base = forward_embed(m, x).argmax(1)
    scale = sigma * np.sqrt(T * d)

    def flip(r):
        rr = torch.as_tensor(np.broadcast_to(np.asarray(r, dtype=np.float64), (n,)).copy(),
                             dtype=x.dtype, device=x.device).reshape(n, 1, 1, 1)
        pert = (x[:, None] + (rr * scale) * g).reshape(n * n_dirs, T, d)
        with torch.no_grad():
            pr = torch.cat([forward_embed(m, pert[i:i + 2048]).argmax(1) for i in range(0, len(pert), 2048)])
        return (pr.reshape(n, n_dirs) != base[:, None]).sum(1).cpu().numpy() >= majority

    return flip, sigma


def shuffle_null(rho, radius, n_perm=1000, seed=0):
    """Permutation p of Spearman(rho, radius) against 1000 shuffles of rho (SPEC_O O1.4 null). (nan, nan) when the
    measurement is degenerate — a constant failure radius, which is itself the finding, not a null result."""
    rng = np.random.default_rng(seed)
    t = _spear(rho, radius)
    if not np.isfinite(t["spearman"]): return float("nan"), float("nan")
    obs = t["spearman"]
    d = np.array([stats.spearmanr(rng.permutation(rho), radius).statistic for _ in range(n_perm)])
    return float(obs), float((np.sum(d <= obs) + 1) / (n_perm + 1))


@guarded
def grok_P5(m, data, name, n_gauge, seed=0):
    rng = np.random.default_rng(seed)
    te, tr = data["te"], data["tr"]
    i_eval = te[rng.permutation(len(te))[:N_P5]]
    i_bg = tr[rng.permutation(len(tr))[:N_BG]]
    x_eval = embed(m, olib.grok_tokens(data, i_eval)).detach()
    x_bg = embed(m, olib.grok_tokens(data, i_bg)).detach()
    flip, sigma = grok_perturbation(m, x_eval, seed=seed)
    radius = grid_radius(flip, len(x_eval))                                 # v1, the pre-registered grid
    radius_v2 = bisect_radius(flip, len(x_eval))                            # v2, AMENDMENTS #O1.4 addenda (v)
    frac_inf = float(np.mean(~np.isfinite(radius)))

    def predictors(model):
        a_ev, l_ev = acts_logits(model, x_eval)
        a_bg, _ = acts_logits(model, x_bg)
        sc = std_scales(np.concatenate([a_ev, a_bg]))
        cur = curvature(np.concatenate([a_ev, a_bg]) / sc[None, :], k=K_CURV)
        lip = jacobian_lip(model, x_eval)
        _, negmargin, margin = logit_baselines(l_ev)
        return cur.norm[:len(a_ev)] * lip, cur.norm[:len(a_ev)], lip, negmargin, cur.dim

    rho, qnorm, lip, negmargin, cdim = predictors(m)
    t_rho = _spear(rho, radius); t_base = _spear(negmargin, radius)
    obs, null_p = shuffle_null(rho, radius, seed=seed)
    beats = bool(np.isfinite(t_rho["spearman"]) and np.isfinite(t_base["spearman"]) and t_rho["spearman"] < t_base["spearman"])
    ok = (t_rho["spearman"] < SPEARMAN_MAX and t_rho["p_value"] < P_MAX and beats and null_p < P_MAX)

    def spear_vec(model):
        r, _, _, _, _ = predictors(model)
        return np.array([_spear(r, radius)["spearman"], _spear(r, radius_v2)["spearman"]])
    orbit = np.asarray(olib.on_orbit(m, spear_vec, n=n_gauge, seed=42))
    cvv = olib.cv(orbit[:, 0]); cv2 = olib.cv(orbit[:, 1])
    v2 = v2_block(rho, negmargin, radius_v2, cv2, seed, "softmax margin")
    status = "certified" if (ok and cvv < CV_MAX) else ("failed (artifact)" if cvv >= CV_MAX else "failed")
    diag = {}
    if frac_inf > 0.5:                                   # DIAGNOSTIC ONLY, never a criterion (AMENDMENTS #O1.4i)
        rx = grid_radius(flip, len(x_eval), RADII_EXT)
        diag = {"note": "the pre-registered radius grid has no dynamic range on this specimen; the extended grid below is "
                        "a diagnostic (AMENDMENTS #O1.4i), NOT the pre-registered criterion",
                "radii": list(RADII_EXT), "frac_inf": float(np.mean(~np.isfinite(rx))),
                "radius_hist": {str(r): int((rx == r).sum()) for r in RADII_EXT},
                "spearman_rho": _spear(rho, rx), "spearman_baseline": _spear(negmargin, rx)}
    return cell(status, value=dict(spearman=t_rho["spearman"], p=t_rho["p_value"], spearman_baseline=t_base["spearman"], null_p=null_p),
                criterion=f"Spearman(rho, failure radius) < {SPEARMAN_MAX} at p < {P_MAX}, more negative than the softmax-margin baseline's, shuffled-rho permutation p < {P_MAX}; K-O-2 CV < {CV_MAX}",
                provenance=("rho = ‖Q_h‖ (geometry.curvature, k=%d, on the per-unit-standardised MLP cloud at '=' — %d eval + %d background points) "
                            "x Lip (spectral norm of the autograd Jacobian of the last-position logits w.r.t. the embedded input, AMENDMENTS #O1.4c); "
                            "failure radius = smallest r in {0.01,...,0.5} (relative, sigma_emb=%.4g) at which >= %d of %d Gaussian draws flip the argmax "
                            "(AMENDMENTS #O1.4d); v2 radius = %d-step bisection on [0, %g] with the same rule (addenda (v), reported under evidence.v2 and the _v2 CSV rows; the cell status is the v1 verdict); baseline = -logit margin; null = 1000 shuffles of rho; O1.4 ships at gauge orbit n=%d") % (K_CURV, len(i_eval), N_BG, sigma, MAJORITY, N_DIRS, BISECT_STEPS, R_MAX_V2, n_gauge),
                spearman=t_rho, baseline=t_base, null_p=null_p, beats_baseline=beats, cv_spearman=cvv, n_gauge=n_gauge,
                frac_inf=frac_inf, n=int(len(radius)), radius_hist={str(r): int((radius == r).sum()) for r in RADII},
                sigma_embedded=sigma, curvature_dim=int(cdim), rho_median=float(np.median(rho)),
                qnorm_median=float(np.median(qnorm)), lip_median=float(np.median(lip)),
                degenerate_radius=bool(frac_inf > 0.9), diagnostics=diag, v2=v2)


def v2_block(rho, baseline, radius_v2, cv2, seed, baseline_name):
    """P5 v2 (AMENDMENTS #O1.4 addenda (v)): the same criteria against the bisection-measured failure radius.
    Reported alongside the v1 cell; it never replaces the v1 verdict."""
    t_rho = _spear(rho, radius_v2); t_base = _spear(baseline, radius_v2)
    _, null_p = shuffle_null(rho, radius_v2, seed=seed)
    beats = bool(np.isfinite(t_rho["spearman"]) and np.isfinite(t_base["spearman"]) and t_rho["spearman"] < t_base["spearman"])
    ok = (t_rho["spearman"] < SPEARMAN_MAX and t_rho["p_value"] < P_MAX and beats and null_p < P_MAX)
    verdict = "certified" if (ok and np.isfinite(cv2) and cv2 < CV_MAX) else ("failed (artifact)" if (np.isfinite(cv2) and cv2 >= CV_MAX) else "failed")
    r = np.asarray(radius_v2, float)
    fin = r[np.isfinite(r)]
    return {"measurement": f"{BISECT_STEPS}-step bisection on [0, {R_MAX_V2}] (relative), same majority-of-{N_DIRS} rule",
            "spearman": t_rho, "baseline": t_base, "baseline_name": baseline_name, "null_p": null_p,
            "beats_baseline": beats, "cv_spearman": cv2, "verdict": verdict,
            "frac_inf": float(np.mean(~np.isfinite(r))), "n": int(r.size),
            "radius_median": float(np.median(fin)) if fin.size else float("nan"),
            "radius_iqr": [float(np.percentile(fin, 25)), float(np.percentile(fin, 75))] if fin.size else None}


def _spear(pred, radius):
    a = np.asarray(pred, float).ravel(); b = np.asarray(radius, float).ravel()
    keep = ~(np.isnan(a) | np.isnan(b)); a, b = a[keep], b[keep]
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return {"spearman": float("nan"), "p_value": float("nan"), "n": int(a.size), "reason": "degenerate (constant predictor or constant radius)"}
    r = stats.spearmanr(a, b)
    return {"spearman": float(r.statistic), "p_value": float(r.pvalue), "n": int(a.size)}


# --------------------------------------------------------------------------------------------- ring gauge group (#4g)
def ring_gauge_apply(model: RingRNN, rng, log_range=(0.1, 10.0)):
    """h -> h S with S = P D (permutation x positive diagonal): W -> S^-1 W S, U -> U S, b -> b S, R -> S^-1 R.
    An exact symmetry of h' = (1-a)h + a relu(hW + uU + b) with h_0 = 0 (ReLU commutes with S because D > 0)."""
    n = model.n
    perm = rng.permutation(n); d = np.exp(rng.uniform(np.log(log_range[0]), np.log(log_range[1]), size=n))
    S = np.zeros((n, n)); S[np.arange(n), perm] = d
    Si = np.zeros((n, n)); Si[perm, np.arange(n)] = 1.0 / d
    new = copy.deepcopy(model)
    with torch.no_grad():
        St = torch.as_tensor(S, dtype=new.W.dtype, device=new.W.device)
        Sit = torch.as_tensor(Si, dtype=new.W.dtype, device=new.W.device)
        new.W.copy_(Sit @ new.W @ St); new.U.copy_(new.U @ St); new.b.copy_(new.b @ St); new.R.copy_(Sit @ new.R)
    return new, St


def ring_on_orbit(model, fn, n=20, seed=0):
    """fn(g.model, S) — S is passed so that query points constructed *once* in the trained gauge (the off-attractor
    kicks) can be transported as h -> hS instead of being redrawn, which would make the orbit measure the sampling
    noise of the random kick rather than gauge invariance."""
    rng = np.random.default_rng(seed)
    return [fn(*ring_gauge_apply(model, rng)) for _ in range(n)]


@torch.no_grad()
def ring_states(model, u):
    """Hidden states (B, T, n). Rolled out here rather than through RingRNN.forward, whose h_0 is allocated at the
    default dtype: the gauge orbit is run in float64 (a float32 round-off in a pre-activation flips a ReLU gate and
    the error compounds over 100 recurrent steps, which would show up as a spurious K-O-2 artifact)."""
    B, T, _ = u.shape
    h = torch.zeros(B, model.n, device=u.device, dtype=u.dtype)
    hs = []
    for t in range(T):
        h = ring_step(model, h, u[:, t]); hs.append(h)
    return torch.stack(hs, 1)


def ring_readout(model, H):
    return H @ model.R + model.rb


# --------------------------------------------------------------------------------------------- P4 on the ring specimens
@guarded
def ring_P4(model, name, n_gauge, seed=0, sigma_train=0.3):
    rng = np.random.default_rng(1000 + seed)
    dt = model.W.dtype
    u_c, _, _ = ring_batch(rng, 32, 100, sigma_train); u_c = u_c.to(dt)
    u_h, _, _ = ring_batch(rng, 16, 100, sigma_train); u_h = u_h.to(dt)
    u_f, _, _ = ring_batch(rng, 16, 100, 3 * sigma_train); u_f = u_f.to(dt)      # 3x the training velocity sigma
    u_k, _, _ = ring_batch(rng, 16, 100, sigma_train); u_k = u_k.to(dt)

    def pick(H, n, tag):
        """Deterministic subsample (the same point indices for every gauge sample, so the orbit is comparable)."""
        X = H[:, 20:].reshape(-1, H.shape[-1])                      # drop the transient
        i = np.random.default_rng(1_000 * seed + tag).permutation(len(X))[:n]
        return X[i]

    fixed = {}

    def build(model, S=None):
        cloud = pick(ring_states(model, u_c), N_CLOUD, 1)
        held = pick(ring_states(model, u_h), N_QUERY, 2)
        fast = pick(ring_states(model, u_f), N_QUERY, 3)
        base = pick(ring_states(model, u_k), N_QUERY, 4)
        if "kick" not in fixed:                                     # built once, in the trained gauge
            s_unit = torch.as_tensor(std_scales(cloud.detach().cpu().numpy().astype(np.float64)), dtype=base.dtype, device=base.device)
            z = torch.as_tensor(np.random.default_rng(seed + 7).standard_normal(tuple(base.shape)), dtype=base.dtype, device=base.device)
            fixed["kick"] = base + z * s_unit[None, :]              # per-unit kick: isotropic in the standardised cloud
        kick = fixed["kick"] if S is None else fixed["kick"] @ S    # transported, not redrawn
        return cloud, {"heldout": held, "ood_velocity_3sigma": fast, "ood_kick_1sigma": kick}, float(cloud.std())

    def scores(model, S=None):
        cloud, conds, sg = build(model, S)
        order = list(conds); Q = torch.cat([conds[c] for c in order]); sizes = [len(conds[c]) for c in order]
        C = cloud.detach().cpu().numpy().astype(np.float64); Qn = Q.detach().cpu().numpy().astype(np.float64)
        sc = std_scales(C)
        res = out_of_manifold(C / sc[None, :], Qn / sc[None, :], k=15, t=1, n_eigs=10, seed=seed)
        nul = random_projection_null(C / sc[None, :], Qn / sc[None, :], d=10, seed=seed)
        with torch.no_grad(): nrm = ring_readout(model, Q).norm(dim=1).detach().cpu().numpy()
        return res, nul, np.abs(nrm - 1.0), order, sizes, sg

    res, nul, conf, order, sizes, sg = scores(model)
    cut = np.cumsum([0] + sizes); sl = {c: slice(cut[i], cut[i + 1]) for i, c in enumerate(order)}
    ood = [c for c in order if c != "heldout"]; inn = np.arange(cut[0], cut[1])
    pooled = np.concatenate([np.arange(cut[i], cut[i + 1]) for i, c in enumerate(order) if c != "heldout"])

    def table(scr):
        t = {c: auc_of(scr[inn], scr[sl[c]]) for c in ood}; t["pooled"] = auc_of(scr[inn], scr[pooled]); return t

    A, Acon, Anul = table(res.d_M), table(conf), table(nul.d_M)
    beats = {c: bool(A[c] > Acon[c]) for c in A}; beats_null = {c: bool(A[c] > Anul[c]) for c in A}
    inverted = {c: bool(A[c] < 0.5) for c in A}
    ok = A["pooled"] > AUC_MIN and beats["pooled"] and beats_null["pooled"]
    orbit = ring_on_orbit(model, lambda mm, S: np.array([table(scores(mm, S)[0].d_M)[c] for c in ["pooled"] + ood]), n=n_gauge, seed=43)
    cvv = olib.cv(np.asarray(orbit))
    status = "certified" if (ok and cvv < CV_MAX) else ("failed (artifact)" if cvv >= CV_MAX else "failed")
    diag = {}
    if status != "certified":
        cloud0, conds0, _ = build(model)
        C0 = cloud0.detach().cpu().numpy().astype(np.float64); sc0 = std_scales(C0)
        Q0 = torch.cat([conds0[c] for c in order]).detach().cpu().numpy().astype(np.float64)
        diag = nn_diagnostics(C0 / sc0[None, :], Q0 / sc0[None, :], inn, sl, ood, pooled)
    return cell(status, value=dict(auc=A["pooled"], auc_readout_norm=Acon["pooled"], auc_null=Anul["pooled"]),
                criterion=f"AUC(d_M, held-out states vs pooled OOD states) > {AUC_MIN} AND > the readout-norm baseline AND > the random-projection null; K-O-2 CV < {CV_MAX}",
                provenance=("spectral.out_of_manifold on the hidden-state cloud (t >= 20, transient dropped); in-distribution = held-out sequences at the "
                            f"training sigma=0.3; OOD = sequences at 3 sigma and in-distribution states kicked by N(0, s_j^2) in unit j, s_j the in-distribution per-unit std (equivariant, so the kick is isotropic in the standardised cloud); "
                            "cloud standardised per unit (gauge fixing); baseline = |‖R h + rb‖ - 1| (AMENDMENTS #O1.4e); "
                            f"gauge orbit n={n_gauge} over the ring gauge group (AMENDMENTS #O1.4g)"),
                auc=A, auc_readout_norm=Acon, auc_null=Anul, beats_baselines=beats, beats_null=beats_null,
                n_in=int(len(inn)), n_ood={c: int(sizes[order.index(c)]) for c in ood}, cv_auc=cvv, n_gauge=n_gauge,
                threshold=float(res.threshold), frac_flagged_heldout=float(res.flags[inn].mean()),
                frac_flagged={c: float(res.flags[sl[c]].mean()) for c in ood}, state_sigma=sg, inverted=inverted, diagnostics=diag)


# --------------------------------------------------------------------------------------------- P5 on the ring specimens
def ring_step(model, h, u_t):
    return (1 - model.alpha) * h + model.alpha * torch.relu(h @ model.W + u_t @ model.U + model.b)


def ring_jac_norm(model, H, U, s_unit):
    """‖dh_{t+1}/dh_t‖_2 with mask = 1[pre-activation > 0], measured in the per-unit-standardised coordinates
    h~ = h / s: J_std = diag(s) [(1-a) I + a (W * mask)] diag(1/s). In raw coordinates J -> S^-1 J S under the ring
    gauge group (AMENDMENTS #O1.4g) and its spectral norm is NOT invariant; in the standardised coordinates the gauge
    acts by permutation conjugation only, so ‖J_std‖_2 is."""
    with torch.no_grad():
        pre = H @ model.W + U @ model.U + model.b
        mask = (pre > 0).to(H.dtype)                                   # (n_pts, n)
        I = torch.eye(model.n, device=H.device, dtype=H.dtype)
        s = s_unit.to(H.dtype)
        out = []
        for i in range(0, len(H), 128):
            J = (1 - model.alpha) * I[None] + model.alpha * (model.W[None] * mask[i:i + 128, None, :])
            J = s[None, :, None] * J * (1.0 / s)[None, None, :]
            G = torch.bmm(J, J.transpose(1, 2))
            out.append(torch.sqrt(torch.clamp(torch.linalg.eigvalsh(G)[:, -1], min=0.0)).cpu().numpy())
    return np.concatenate(out)


@guarded
def ring_P5(model, name, n_gauge, seed=0, sigma_train=0.3, t0=50):
    rng = np.random.default_rng(2000 + seed)
    dt = model.W.dtype
    u, _, th = ring_batch(rng, N_P5, t0 + RING_ROLL + 1, sigma_train); u = u.to(dt)
    u_bg, _, _ = ring_batch(rng, 16, 100, sigma_train); u_bg = u_bg.to(dt)
    with torch.no_grad():
        H = ring_states(model, u)
        h0 = H[:, t0]                                                  # (n, 128) the states rho is evaluated at
        H_bg = ring_states(model, u_bg)[:, 20:].reshape(-1, model.n)
        i_bg = np.random.default_rng(seed).permutation(len(H_bg))[:N_BG]
        H_bg = H_bg[i_bg]
        # unperturbed reference: roll 50 further steps
        h = h0.clone()
        for t in range(t0 + 1, t0 + 1 + RING_ROLL): h = ring_step(model, h, u[:, t])
        o_ref = ring_readout(model, h); ref_ang = torch.atan2(o_ref[:, 1], o_ref[:, 0]).cpu().numpy()
    s_ref = std_scales(H_bg.detach().cpu().numpy().astype(np.float64))
    s_unit = torch.as_tensor(s_ref, dtype=h0.dtype, device=h0.device)
    sg = float(np.mean(s_ref))
    g = torch.as_tensor(np.random.default_rng(seed).standard_normal((len(h0), N_DIRS, model.n)), dtype=h0.dtype, device=h0.device)
    g = g / g.norm(dim=2, keepdim=True)
    g = g * s_unit[None, None, :]                                      # per-unit scale: equivariant under the ring gauge group
    uu = u.repeat_interleave(N_DIRS, 0)
    n_pt = len(h0)

    def flip(r):
        """r a per-input relative radius -> bool (n,), True where >= MAJORITY of the 8 draws leave the readout more
        than RING_FAIL_RAD off the unperturbed model after RING_ROLL further steps."""
        rr = torch.as_tensor(np.broadcast_to(np.asarray(r, dtype=np.float64), (n_pt,)).copy(),
                             dtype=h0.dtype, device=h0.device).reshape(n_pt, 1, 1)
        with torch.no_grad():
            hp = (h0[:, None] + (rr * np.sqrt(model.n)) * g).reshape(-1, model.n)
            for t in range(t0 + 1, t0 + 1 + RING_ROLL): hp = ring_step(model, hp, uu[:, t])
            o = ring_readout(model, hp); ang = torch.atan2(o[:, 1], o[:, 0]).cpu().numpy().reshape(-1, N_DIRS)
        err = np.abs(np.angle(np.exp(1j * (ang - ref_ang[:, None]))))
        return (err > RING_FAIL_RAD).sum(1) >= MAJORITY

    def sweep(radii):
        return grid_radius(flip, n_pt, radii)

    radius = sweep(RADII)                                                   # v1, the pre-registered grid
    radius_v2 = bisect_radius(flip, n_pt)                                   # v2, AMENDMENTS #O1.4 addenda (v)
    frac_inf = float(np.mean(~np.isfinite(radius)))

    def predictors(mm):
        with torch.no_grad():
            Hh = ring_states(mm, u)[:, t0]
            Hb = ring_states(mm, u_bg)[:, 20:].reshape(-1, mm.n)[i_bg]
            X = torch.cat([Hh, Hb]).detach().cpu().numpy().astype(np.float64)
            sc = std_scales(X)
            cur = curvature(X / sc[None, :], k=K_CURV)
            lipj = ring_jac_norm(mm, Hh, u[:, t0], torch.as_tensor(sc, dtype=Hh.dtype, device=Hh.device))
            base = np.abs(ring_readout(mm, Hh).norm(dim=1).detach().cpu().numpy() - 1.0)
        return cur.norm[:len(Hh)] * lipj, cur.norm[:len(Hh)], lipj, base, cur.dim

    rho, qnorm, lipj, base, cdim = predictors(model)
    t_rho = _spear(rho, radius); t_base = _spear(base, radius)
    obs, null_p = shuffle_null(rho, radius, seed=seed)
    beats = bool(np.isfinite(t_rho["spearman"]) and np.isfinite(t_base["spearman"]) and t_rho["spearman"] < t_base["spearman"])
    ok = (t_rho["spearman"] < SPEARMAN_MAX and t_rho["p_value"] < P_MAX and beats and null_p < P_MAX)
    def spear_vec(mm, S):
        r = predictors(mm)[0]
        return np.array([_spear(r, radius)["spearman"], _spear(r, radius_v2)["spearman"]])
    orbit = np.asarray(ring_on_orbit(model, spear_vec, n=n_gauge, seed=44))
    cvv = olib.cv(orbit[:, 0]); cv2 = olib.cv(orbit[:, 1])
    v2 = v2_block(rho, base, radius_v2, cv2, seed, "readout-norm deviation")
    status = "certified" if (ok and cvv < CV_MAX) else ("failed (artifact)" if cvv >= CV_MAX else "failed")
    diag = {}
    if frac_inf > 0.5:                                   # DIAGNOSTIC ONLY, never a criterion (AMENDMENTS #O1.4i)
        rx = sweep(RADII_EXT)
        diag = {"note": "the pre-registered radius grid has no dynamic range on this specimen; the extended grid below is "
                        "a diagnostic (AMENDMENTS #O1.4i), NOT the pre-registered criterion",
                "radii": list(RADII_EXT), "frac_inf": float(np.mean(~np.isfinite(rx))),
                "radius_hist": {str(r): int((rx == r).sum()) for r in RADII_EXT},
                "spearman_rho": _spear(rho, rx), "spearman_baseline": _spear(base, rx)}
    return cell(status, value=dict(spearman=t_rho["spearman"], p=t_rho["p_value"], spearman_baseline=t_base["spearman"], null_p=null_p),
                criterion=f"Spearman(rho, failure radius) < {SPEARMAN_MAX} at p < {P_MAX}, more negative than the readout-norm baseline's, shuffled-rho permutation p < {P_MAX}; K-O-2 CV < {CV_MAX}",
                provenance=("rho = ‖Q_h‖ (geometry.curvature, k=%d, per-unit-standardised state cloud: %d eval states at t=%d + %d background states) "
                            "x ‖dh_{t+1}/dh_t‖_2; failure = angular error > %.2f rad vs the unperturbed model after %d further steps on >= %d of %d Gaussian "
                            "draws, perturbation scaled per unit by the in-distribution per-unit std (mean %.4g), so the draw is isotropic in the standardised cloud; baseline = |‖readout‖-1| (AMENDMENTS #O1.4e); null = 1000 shuffles of rho; "
                            "v2 radius = %d-step bisection on [0, %g] with the same rule (addenda (v), evidence.v2 / the _v2 CSV rows; the cell status is the v1 verdict); O1.4 ships at gauge orbit n=%d over the ring gauge group (AMENDMENTS #O1.4g)") % (K_CURV, len(h0), t0, N_BG, RING_FAIL_RAD, RING_ROLL, MAJORITY, N_DIRS, sg, BISECT_STEPS, R_MAX_V2, n_gauge),
                spearman=t_rho, baseline=t_base, null_p=null_p, beats_baseline=beats, cv_spearman=cvv, n_gauge=n_gauge,
                frac_inf=frac_inf, n=int(len(radius)), radius_hist={str(r): int((radius == r).sum()) for r in RADII},
                state_sigma=sg, curvature_dim=int(cdim), rho_median=float(np.median(rho)),
                qnorm_median=float(np.median(qnorm)), jac_median=float(np.median(lipj)),
                degenerate_radius=bool(frac_inf > 0.9), diagnostics=diag, v2=v2)


# --------------------------------------------------------------------------------------------- driver
def _append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); new and w.writeheader()
        for r in rows: w.writerow(r)


def p4_rows(name, kind, c):
    ev = c.get("evidence", {}); rows = []
    if not ev.get("auc"):
        return [dict(specimen=name, kind=kind, condition="pooled", scorer="diffusion", auc="", n_in="", n_ood="",
                     beats_baselines="", beats_null="", status=c["status"], cv_auc="", n_gauge="", wall=c.get("wall"))]
    base_keys = [("diffusion", "auc"), ("max_softmax", "auc_maxsoftmax"), ("logit_margin", "auc_margin"),
                 ("readout_norm", "auc_readout_norm"), ("random_projection_null", "auc_null")]
    for cond in ev["auc"]:
        for scorer, key in base_keys:
            if key not in ev: continue
            rows.append(dict(specimen=name, kind=kind, condition=cond, scorer=scorer, auc=f"{ev[key][cond]:.4f}",
                             n_in=ev.get("n_in"), n_ood=ev.get("n_ood", {}).get(cond, ""),
                             beats_baselines=int(ev["beats_baselines"][cond]) if scorer == "diffusion" else "",
                             beats_null=int(ev["beats_null"][cond]) if scorer == "diffusion" else "",
                             status=c["status"] if cond == "pooled" else "", cv_auc=f"{ev.get('cv_auc', float('nan')):.3e}",
                             n_gauge=ev.get("n_gauge"), wall=c.get("wall")))
    return rows


def _p5_triplet(name, kind, ev, wall, sfx, status, spear, base, null_p, frac_inf, beats, cv):
    """Three rows (predictor, baseline, shuffled-rho null) for one failure-radius measurement."""
    rows = []
    for pred, d in ((f"rho{sfx}", spear), (f"baseline_margin_or_readout{sfx}", base)):
        rows.append(dict(specimen=name, kind=kind, predictor=pred, spearman=f"{d['spearman']:.4f}", p_value=f"{d['p_value']:.3e}",
                         n=d["n"], frac_inf=f"{frac_inf:.3f}", beats_baseline=int(beats) if pred.startswith("rho") else "",
                         null_p=f"{null_p:.4f}" if pred.startswith("rho") else "", status=status if pred.startswith("rho") else "",
                         cv_spearman=f"{cv:.3e}", n_gauge=ev.get("n_gauge"), wall=wall))
    rows.append(dict(specimen=name, kind=kind, predictor=f"rho_shuffled_null{sfx}", spearman="", p_value="", n=spear["n"],
                     frac_inf="", beats_baseline="", null_p=f"{null_p:.4f}", status="", cv_spearman="",
                     n_gauge=ev.get("n_gauge"), wall=wall))
    return rows


def p5_rows(name, kind, c):
    """v1 rows (the pre-registered grid) and, beside them, the `_v2` rows (bisection, AMENDMENTS #O1.4 addenda (v))."""
    ev = c.get("evidence", {}); wall = c.get("wall")
    if "spearman" not in ev:
        return [dict(specimen=name, kind=kind, predictor="rho", spearman="", p_value="", n="", frac_inf="",
                     beats_baseline="", null_p="", status=c["status"], cv_spearman="", n_gauge="", wall=wall)]
    rows = _p5_triplet(name, kind, ev, wall, "", c["status"], ev["spearman"], ev["baseline"], ev["null_p"],
                       ev["frac_inf"], ev["beats_baseline"], ev.get("cv_spearman", float("nan")))
    v2 = ev.get("v2")
    if v2:
        rows += _p5_triplet(name, kind, ev, wall, "_v2", v2["verdict"], v2["spearman"], v2["baseline"], v2["null_p"],
                            v2["frac_inf"], v2["beats_baseline"], v2["cv_spearman"])
    return rows


def run(name, kind, n_gauge, cells, force=False):
    jpath = os.path.join(OUT, f"fine_motion_{name}.json")
    if os.path.exists(jpath) and not force:
        print(f"skip {name} (done)", flush=True); return json.load(open(jpath))
    out = dict(specimen=name, kind=kind, n_gauge=n_gauge, cells={}, spec="SPEC_O O1.4 / FINAL_SPEC 6.5-6.6; realisation AMENDMENTS #O1.4")
    if kind == "grok":
        m, ck, data = olib.load_grok(name)
        out["condition"] = ck["cfg"]["condition"]; out["seed"] = ck["cfg"]["seed"]
        out["test_acc"] = ck.get("test_acc")
        if "P4" in cells: out["cells"]["P4"] = grok_P4(m, data, name, n_gauge)
        if "P5" in cells: out["cells"]["P5"] = grok_P5(m, data, name, n_gauge)
    else:
        model, ck = olib.load_ring(name); model = model.double()   # exact gauge orbit: see ring_gauge_apply
        out["condition"] = "ring"; out["seed"] = ck["cfg"]["seed"]; out["ang_err"] = ck.get("ang_err")
        if "P4" in cells: out["cells"]["P4"] = ring_P4(model, name, n_gauge)
        if "P5" in cells: out["cells"]["P5"] = ring_P5(model, name, n_gauge)
    json.dump(out, open(jpath, "w"), indent=1, default=str)
    if "P4" in out["cells"]: _append(P4_CSV, P4_FIELDS, p4_rows(name, kind, out["cells"]["P4"]))
    if "P5" in out["cells"]: _append(P5_CSV, P5_FIELDS, p5_rows(name, kind, out["cells"]["P5"]))
    for k, c in out["cells"].items():
        print(f"{name:18s} {k} {c['status']:18s} {json.dumps(c['value'], default=str)[:150]}  [{c.get('wall')}s]", flush=True)
        if c["status"] == "not_reached": print("   ", c["provenance"], flush=True)
    return out


def included_specimens():
    rows = list(csv.DictReader(open(os.path.join(RES, "specimens.csv"))))
    return [(r["name"], r["kind"]) for r in rows if r["included"] == "1"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimens", nargs="*")
    ap.add_argument("--cells", nargs="*", default=["P4", "P5"])
    ap.add_argument("--n-gauge", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    known = dict(included_specimens())
    names = [(n, known.get(n, "ring" if n.startswith("ring") else "grok")) for n in a.specimens] if a.specimens else included_specimens()
    for n, k in names:
        run(n, k, a.n_gauge, a.cells, a.force)
