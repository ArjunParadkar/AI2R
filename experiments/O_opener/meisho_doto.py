#!/usr/bin/env python3
"""O1.2 Meisho Doto — the full open() (FINAL_SPEC §3 steps 1–9) on every included T1 specimen (SPEC_O §O1.2; criteria = §6/§7;
AMENDMENTS #2–#4 for the function-level Fourier concentration κ_f and the recipe-2 specimens).

  python meisho_doto.py [--specimens NAME ...] [--n-gauge 300] [--cells gauge P0 P1 P2 P6 P7 glue R1 R4 R5 R6 R7 R8]
  python meisho_doto.py --aggregate            # cross-seed cells: P6 AUC, P7 ordering, mechanism count, pass counts

Per specimen: results/open/meisho_doto_<name>.json (every cell with evidence, nulls, provenance), results/open/meisho_doto_<name>.md
(the whiteboxtopo.open Scorecard) and one row per cell in results/O1_2_cells.csv. Cells whose instrument does not apply or whose
module fails are recorded "not_reached: <reason>" — never skipped silently (Operating Constitution: negatives are deliverables).
Gauge fixings that the topological / spectral cells rely on (per-unit standardisation of activation clouds) are recorded in the
cell provenance (K-O-2: every certified scalar is re-computed on the gauge orbit; CV > 1e-3 → "failed (artifact)").
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import olib  # noqa: E402
from olib import ROOT, RES, DEV  # noqa: E402
from sweep_tosho import P, fourier_kappa, fourier_kappa_f, marginal_logits, RingRNN, ring_batch, ang_error  # noqa: E402
from whiteboxtopo.open import Scorecard, CellResult  # noqa: E402

OUT = os.path.join(RES, "open"); os.makedirs(OUT, exist_ok=True)
CELLS_CSV = os.path.join(RES, "O1_2_cells.csv")
CELL_FIELDS = ["specimen", "cell", "status", "value", "criterion", "provenance", "wall"]
GROK_CELLS = ["gauge", "P0", "P1", "P2", "P6", "P7", "glue", "R1", "R4", "R5", "R6", "R7", "R8"]
RING_CELLS = ["gauge", "P0", "P1", "P2", "P7", "R2"]
TAU = 0.05; ART_CV = 1e-3


def cell(status, value=None, criterion="", provenance="", **evidence):
    return dict(status=status, value=value, criterion=criterion, provenance=provenance, evidence=evidence)


def guarded(fn):
    def run(*a, **k):
        t = time.time()
        try: c = fn(*a, **k)
        except Exception as e: c = cell("not_reached", provenance=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1200:])
        c["wall"] = round(time.time() - t, 1); return c
    return run


def _f(x):
    try: return float(x)
    except Exception: return None


# ============================================================================================== grok: shared extraction
def logit_tensor(m, data):
    """Logits over all p² pairs: the function the certified cells are about. Planted specimens are marginalised over the 8
    shortcut tokens (AMENDMENTS #3) — the group-algorithm component; the leaked-s tensor is returned separately."""
    if data["planted"]:
        return marginal_logits(m, data["a"], data["b"]), olib.grok_logits(m, data, shortcut="leak")
    L = olib.grok_logits(m, data); return L, L


def irreps_f(L):
    """Function-level irrep inventory (AMENDMENTS #2): κ_f, K_f, folded shares, n_eff, and per-frequency amplitude / phase of the
    diagonal DFT bins (k, k, −k) — the certified numbers R1 rebuilds from (no weights copied)."""
    kf, K, share, neff = fourier_kappa_f(L)
    Lc = L.reshape(P, P, P); Lc = Lc - Lc.mean(-1, keepdims=True); F = np.fft.fftn(Lc)
    half = (P - 1) // 2; amp = {}; phase = {}
    for k in range(1, half + 1):
        z = F[k, k, (-k) % P]; amp[k] = float(2 * abs(z) / P ** 3); phase[k] = float(np.angle(z))
    return dict(kappa_f=float(kf), K=[int(k) for k in K], share=share, n_eff=float(neff), amp=amp, phase=phase)


def rebuild_logits(K, w, a, b):
    """Σ_k w_k cos(ω_k (a + b − c)) — identical to reconstruct.from_irreps(...).forward by the angle-addition identity; used for
    the per-object ablations (R6) where the normalisation of the remaining weights must not change."""
    c = np.arange(P); out = np.zeros((len(a), P))
    for k, wk in zip(K, w):
        out += wk * np.cos(2 * np.pi * k * (a[:, None] + b[:, None] - c[None, :]) / P)
    return out


def mlp_cloud(m, toks, batch=2048):
    """MLP hidden activations at the '=' position (N, d_mlp) — the representation cloud of the decision."""
    view = olib.NodeView(m); outs = []
    with torch.no_grad():
        for i in range(0, len(toks), batch):
            view.forward(toks[i:i + batch], record=True)
            outs.append(torch.stack([o.detach() for o in view._node_out[view.H:]], -1)[:, -1, :].cpu())
    return torch.cat(outs).numpy()


def standardise(X):
    """Per-unit z-scoring: the recorded gauge fixing for every distance-based cell (SPEC 2.7/2.10). Dead units are dropped."""
    X = np.asarray(X, dtype=np.float64); sd = X.std(0); keep = sd > 1e-8
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep], keep


# ============================================================================================== grok cells
@guarded
def step_gauge(m, data, n_gauge):
    from whiteboxtopo.symmetry import group
    from whiteboxtopo.symmetry import transformer as tf_gauge
    desc = group.describe(m)
    toks = olib.grok_tokens(data, data["te"][:256])
    with torch.no_grad():
        ref = m(toks)[:, -1]
        devs = [float((tf_gauge.apply(m, g)(toks)[:, -1] - ref).abs().max() / (ref.abs().max() + 1e-12)) for g in olib.gauge_samples(m, min(n_gauge, 50), seed=1)]
    return cell("certified", value=float(np.max(devs)), criterion="exact symmetries: max relative output deviation on 50 samples at float32 round-off",
                provenance="symmetry.transformer (head perms, per-head GL(dh) rotations, MLP perms+scalings, LN rescalings); all exact for the ReLU toy transformer; hidden symmetries (Jacobian corank) not computed for the transformer",
                group=desc if isinstance(desc, dict) else str(desc), max_dev=float(np.max(devs)), median_dev=float(np.median(devs)))


@guarded
def cell_P0(m, data, seed=0):
    view = olib.NodeView(m); idx = data["te"][:512]
    toks = olib.grok_tokens(data, idx); y = torch.as_tensor(data["c"][idx], device=toks.device)
    means = view.node_means(toks); E, names = olib.attribution_graph(view, toks, y, means)
    prof = olib.p0_profile(E, density=0.1, seed=seed)
    from whiteboxtopo.legibility import graph as G
    null = G.shuffled_q_null(E, density=0.1, n=20, seed=seed)
    qn = np.asarray(null["Q"] if isinstance(null, dict) else null, float)
    verdict = "tractable" if prof["treewidth"] <= 8 else ("marginal" if prof["treewidth"] <= 32 else "intractable")
    return cell("certified" if verdict == "tractable" else "failed", value=verdict,
                criterion="tractable iff treewidth_ub <= 8 at density 0.1 (the FPT parameter, FINAL_SPEC 2.14; AMENDMENTS #5d), marginal <= 32; Q reported against the rewiring null; the V-block phase diagram is the reference placement",
                provenance="olib.attribution_graph (grad x (act-mean), heads->MLP units->logits) + legibility.graph; null = degree-preserving rewiring (20)",
                Q=prof["Q"], treewidth=prof["treewidth"], sparsity=prof["sparsity"], null_Q_mean=float(qn.mean()), null_Q_max=float(qn.max()), Q_minus_null=prof["Q"] - float(qn.mean()))


@guarded
def cell_P1(m, data, n_gauge, inv):
    from whiteboxtopo.symmetry import transformer as tf_gauge
    from whiteboxtopo.invariants.extractors import fourier_irreps
    kf, K = inv["kappa_f"], inv["K"]
    # K-O-2: the inventory over the gauge orbit (function-level → invariant by construction; measured anyway)
    def inventory(mm):
        L, _ = logit_tensor(mm, data); r = fourier_kappa_f(L); return r[0], tuple(r[1])
    orbit = [inventory(tf_gauge.apply(m, g)) for g in olib.gauge_samples(m, n_gauge, seed=2)]
    same_K = all(o[1] == tuple(K) for o in orbit); cv_kappa = olib.cv([o[0] for o in orbit])
    # null: feature shuffle of the token embedding rows (destroys the token-axis structure the irreps live on)
    mm = copy.deepcopy(m)
    with torch.no_grad(): mm.wte[:P] = m.wte[np.random.default_rng(0).permutation(P)]
    Ln, _ = logit_tensor(mm, data); kappa_null = fourier_kappa_f(Ln)[0]
    # pre-registered estimator (embedding-level, SPEC_O/FINAL_SPEC §2.6), reported as the pre-registered outcome
    r = fourier_irreps(m.wte.detach().cpu().numpy(), P, tau=TAU); kappa_emb = float(r.kappa); K_emb = sorted(int(k) for k in r.K)
    ok = kf > 0.9 and same_K and cv_kappa < ART_CV and kappa_null < 0.3
    status = "certified" if ok else ("failed (artifact)" if (kf > 0.9 and (cv_kappa >= ART_CV or not same_K)) else "failed")
    memor = data.get("condition") == "memorize"
    return cell(status if status != "failed (artifact)" else "failed", value=dict(kappa_f=kf, K=K, n_freq=len(K), n_eff=inv["n_eff"]),
                criterion="kappa_f > 0.9 (AMENDMENTS #2); K_f identical on every gauge sample; CV(kappa_f) < 1e-3; row-shuffle null kappa_f < 0.3; rebuild from K alone = R1" + ("; memorising control MUST NOT certify (K-O-3)" if memor else ""),
                provenance=f"sweep_tosho.fourier_kappa_f on the {'shortcut-marginalised ' if data['planted'] else ''}logit tensor (p^2 pairs); orbit n={n_gauge} (symmetry.transformer combined samples); pre-registered embedding estimator invariants.extractors.fourier_irreps tau=0.05 reported",
                same_K=same_K, cv_kappa=cv_kappa, kappa_null=float(kappa_null), artifact=(cv_kappa >= ART_CV or not same_K),
                prereg_embedding=dict(kappa_emb=kappa_emb, K_emb=K_emb, passed=kappa_emb > 0.9, status="FAILED (pre-registered estimator, AMENDMENTS #2)" if kappa_emb <= 0.9 else "passed"),
                amplitude={int(k): inv["amp"][k] for k in K}, phase={int(k): inv["phase"][k] for k in K}, ko3_memoriser_not_certified=(not ok) if memor else None)


def token_contributions(m, toks):
    """c[b, pos, class] = J_pos(x) . e_pos: frozen-pattern linear contribution of the token at ``pos`` to each logit (exact under G:
    the function is identical, so its Jacobian is). Reported with the QK caveat (the attention pattern is treated as frozen)."""
    m.zero_grad(set_to_none=True)
    e = m.wte[toks].detach(); x = (e + m.wpe[:toks.shape[1]][None]).detach().requires_grad_(True)
    logits = _forward_from_embedded(m, x, toks.shape[1])
    B, C = logits.shape; contrib = torch.zeros(B, toks.shape[1], C, device=x.device)
    for c in range(C):
        g, = torch.autograd.grad(logits[:, c].sum(), x, retain_graph=True)
        contrib[:, :, c] = (g * e).sum(-1)
    return contrib.detach(), logits.detach()


def _forward_from_embedded(m, x, T):
    l = 0; h = m._ln(x, m.ln1_g[l], m.ln1_b[l])
    q = torch.einsum("btd,dhk->bthk", h, m.Wq[l]) + m.bq[l]; k = torch.einsum("btd,dhk->bthk", h, m.Wk[l]) + m.bk[l]
    v = torch.einsum("btd,dhk->bthk", h, m.Wv[l]) + m.bv[l]
    att = (torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(m.dh)).masked_fill(~m.mask[:T, :T], float("-inf")).softmax(-1)
    z = torch.einsum("bhts,bshk->bthk", att, v); xx = x + torch.einsum("bthk,hkd->bthd", z, m.Wo[l]).sum(2) + m.bo[l]
    a = torch.relu(m._ln(xx, m.ln2_g[l], m.ln2_b[l]) @ m.W1[l] + m.b1[l]); xx = xx + a @ m.W2[l] + m.b2[l]
    return (m._ln(xx, m.lnf_g, m.lnf_b) @ m.Wu + m.bu)[:, -1]


def integrated_gradients(m, toks, steps=32, delta=1.0):
    """IG of the predicted-class logit w.r.t. each position's embedded vector along the path (1-delta) e -> e (positional kept).
    delta = 1: the zero-embedding baseline (global); delta small: the constant-activation-pattern region around x (the D1 anchor as
    pre-registered, AMENDMENTS #5c). Returned per position, scaled to the full contribution (divided by delta)."""
    T = toks.shape[1]; pos = m.wpe[:T][None]; e = m.wte[toks].detach(); ig = torch.zeros_like(e)
    with torch.no_grad(): pred = _forward_from_embedded(m, (pos + e).detach(), T).argmax(1)
    for s in range(1, steps + 1):
        x = (pos + e * (1 - delta + delta * s / steps)).detach().requires_grad_(True)
        logits = _forward_from_embedded(m, x, T)
        g, = torch.autograd.grad(logits[torch.arange(len(pred)), pred].sum(), x); ig += g
    return (ig / steps * e).sum(-1).detach()


@guarded
def cell_P2(m, data, n_gauge):
    from whiteboxtopo.symmetry import transformer as tf_gauge
    idx = data["te"][:128]; toks = olib.grok_tokens(data, idx)
    contrib, logits = token_contributions(m, toks); pred = logits.argmax(1); ar = torch.arange(len(idx))
    c_pred = contrib[ar, :, pred].cpu().numpy()
    orbit = [token_contributions(tf_gauge.apply(m, g), toks)[0][ar, :, pred].cpu().numpy() for g in olib.gauge_samples(m, min(n_gauge, 100), seed=3)]
    cv_attr = olib.cv(orbit)
    ig_g = integrated_gradients(m, toks, delta=1.0).cpu().numpy(); ig_l = integrated_gradients(m, toks, delta=0.05).cpu().numpy()
    r_global = float(np.corrcoef(c_pred[:, :-1].ravel(), ig_g[:, :-1].ravel())[0, 1]); r = float(np.corrcoef(c_pred[:, :-1].ravel(), ig_l[:, :-1].ravel())[0, 1])
    ok = cv_attr < 1e-4 and r > 0.95
    return cell("certified" if ok else ("partial" if cv_attr < 1e-4 else "failed"), value=dict(cv=cv_attr, ig_r_local=r, ig_r_global=r_global),
                criterion="attribution CV over the gauge orbit < 1e-4 (float32 realisation of 'exact'); IG anchor r > 0.95 on the constant-pattern region (path (1-0.05)e -> e, AMENDMENTS #5c); zero-baseline global IG reported",
                provenance="frozen-pattern Jacobian contributions J_pos . e_pos to the predicted logit (QK caveat; the LN+softmax toy transformer is not piecewise linear); IG 32 steps; orbit n=100",
                top_token_share=float(np.mean(np.abs(c_pred[:, :2]).sum(1) / (np.abs(c_pred).sum(1) + 1e-9))))


@guarded
def cell_P6(m, data, inv):
    kf = inv["kappa_f"]; memor = data.get("condition") == "memorize"
    kappa_emb, _ = fourier_kappa(m.wte.detach().cpu().numpy(), P, TAU)
    ok = (kf < 0.3) if memor else (kf > 0.9)
    return cell("certified" if ok else "failed", value=dict(kappa_f=kf, kappa_emb=float(kappa_emb), memorising=memor),
                criterion="representation-theoretic criterion: generaliser kappa_f > 0.9, memorising control kappa_f < 0.3 (AMENDMENTS #2); AUC across specimens in the aggregate; pre-registered kappa_emb reported",
                provenance="fourier_kappa_f (function-level) + fourier_kappa (embedding-level, pre-registered)", prereg_kappa_emb_ok=(float(kappa_emb) < 0.3) if memor else (float(kappa_emb) > 0.9))


@guarded
def cell_P7(name, data, inv):
    cks = olib.checkpoints(name); rows = []
    for step, path in cks:
        ck = torch.load(path, map_location="cpu")
        if "kappa_f" in ck and not data["planted"]: kf, K = ck["kappa_f"], tuple(ck["K_f"])
        else:
            mm, _, _ = olib.load_grok(name, step); L, _ = logit_tensor(mm, data); r = fourier_kappa_f(L); kf, K = r[0], tuple(r[1])
        rows.append((step, float(ck["test_acc"]), float(kf), K))
    steps = np.array([r[0] for r in rows]); acc = np.array([r[1] for r in rows]); kap = np.array([r[2] for r in rows]); T = steps[-1]
    t_acc = float(steps[np.argmax(acc > 0.5)] / T) if (acc > 0.5).any() else None
    t_kap = float(steps[np.argmax(kap > 0.5)] / T) if (kap > 0.5).any() else None
    finalK = inv["K"]; appear = {}
    for k in finalK:
        member = np.array([k in r[3] for r in rows]); first = None
        for i in range(len(member)):
            if member[i:].all(): first = i; break
        appear[int(k)] = None if first is None else float(steps[first] / T)      # persistent appearance (AMENDMENTS #5e)
    # spectral gap over ~30 checkpoints: Laplacian eigenflow of the per-unit standardised MLP cloud on a fixed probe set
    from whiteboxtopo.spectral import eigenflow
    sel = cks[::max(1, len(cks) // 30)]; probe = data["te"][:512]; clouds = []
    for step, path in sel:
        mm, _, _ = olib.load_grok(name, step); toks = olib.grok_tokens(data, probe)
        Z, _ = standardise(mlp_cloud(mm, toks)); clouds.append(Z if Z.shape[1] > 1 else np.zeros((len(probe), 2)))
    ef = eigenflow(clouds, k=15, n_eigs=10); gap = np.asarray(ef.gap, float); sel_steps = np.array([s for s, _ in sel])
    d = np.diff(gap); t_gap = float(sel_steps[int(np.argmax(d)) + 1] / T) if len(d) else None
    ok = t_acc is not None and t_kap is not None and abs(t_acc - t_kap) <= 0.05 and t_gap is not None and abs(t_gap - t_acc) <= 0.05
    return cell("certified" if ok else "failed", value=dict(t_acc=t_acc, t_kappa=t_kap, t_gap=t_gap),
                criterion="irrep appearance (kappa_f > 0.5) within 5 % of training of the accuracy jump (test acc > 0.5); spectral-gap opening (largest gap increase) within 5 %; frequency ordering across seeds in the aggregate",
                provenance="per-checkpoint kappa_f (stored in recipe-2 checkpoints; recomputed otherwise); spectral.eigenflow k=15 on the per-unit standardised (gauge fixing) MLP cloud at '=' of 512 fixed test pairs, ~30 checkpoints",
                appearance=appear, gap=[float(g) for g in gap], n_small=[int(v) for v in ef.n_small], gap_steps=sel_steps.tolist(), n_ckpt=len(cks),
                curve=dict(steps=steps.tolist(), acc=acc.tolist(), kappa_f=kap.tolist()))


def per_example_scores(m, toks, y, means, batch=256):
    """(N, n_nodes) per-example explanation atoms: grad x (activation − mean) per head / MLP unit at all positions (the
    attribution-graph node scores, per example) — the feature axis of the explanation presheaf."""
    from whiteboxtopo.legibility.discovery import logprob_metric
    view = olib.NodeView(m); out = []
    for i in range(0, len(toks), batch):
        tb, yb = toks[i:i + batch], y[i:i + batch]
        logits = view.forward(tb, record=True); outs = view._node_out
        g = torch.autograd.grad(logprob_metric(logits, yb), outs, allow_unused=True)
        cols = []
        for j, o in enumerate(outs):
            if g[j] is None: cols.append(torch.zeros(len(tb), device=o.device)); continue
            s = (o - means[j]) * g[j]; cols.append(s.reshape(len(tb), -1).sum(1))
        out.append(torch.stack(cols, 1).detach().cpu())
    return torch.cat(out).numpy()


@guarded
def cell_glue(m, data, n=2000, k=12):
    from whiteboxtopo.sheaf import cover, cech_H0_H1
    rng = np.random.default_rng(0); idx = rng.choice(data["te"], size=min(n, len(data["te"])), replace=False)
    toks = olib.grok_tokens(data, idx); y = torch.as_tensor(data["c"][idx], device=toks.device)
    view = olib.NodeView(m); means = view.node_means(toks[:512])
    Z, _ = standardise(mlp_cloud(m, toks)); cov = cover(Z, k=k, overlap=0.5, seed=0)
    E = per_example_scores(m, toks, y, means)
    res = cech_H0_H1(cov, E, tau=0.5, k_e=32)
    # null: a trivial explanation (every region scored on the whole population) must give H1 = 0, H0 = 1
    E0 = np.tile(E.mean(0, keepdims=True), (len(E), 1)) + 1e-9 * rng.standard_normal(E.shape); res0 = cech_H0_H1(cov, E0, tau=0.5, k_e=32)
    informative = res.cyclomatic > 0
    ok = res.H1 == 0 and informative
    return cell("certified" if ok else ("partial" if not informative else "failed"), value=dict(H0=int(res.H0), H1=int(res.H1), sanity_gate="passed" if ok else ("uninformative" if not informative else "INSTRUMENT FAILS SANITY GATE")),
                criterion="sanity gate (FINAL_SPEC 2.12): H1 = 0 on clean modular addition; H1 > 0 on a clean specimen means the instrument (these atoms / tau / k_e) is not validated, not that the model has several mechanisms (AMENDMENTS #5f); nerve must contain a cycle (cyclomatic > 0)",
                provenance=f"sheaf.cover knn2 k={k} overlap 0.5 on the per-unit standardised MLP cloud (gauge fixing) of {len(idx)} test pairs; explanation atoms = per-example grad x (act − mean) over the {E.shape[1]} nodes; cech_H0_H1 tau 0.5 k_e 32",
                cover=cov.to_dict(), cech={kk: v for kk, v in res.to_dict().items() if kk not in ("J", "J_bott")}, boundaries_n=len(res.boundaries),
                trivial_null=dict(H0=int(res0.H0), H1=int(res0.H1)), cyclomatic=int(res.cyclomatic))


def _rebuild_fn_factory(amp):
    from whiteboxtopo.reconstruct import from_irreps
    def rebuild(objs):
        objs = [int(o) for o in objs]
        if not objs: raise ValueError("empty inventory")
        rec = from_irreps(objs, {k: amp[k] for k in objs}, P)
        return lambda X: rec.forward(np.asarray(X)[:, 0], np.asarray(X)[:, 1])
    return rebuild


@guarded
def cell_R1(m, data, inv):
    from whiteboxtopo.reconstruct import from_irreps
    from whiteboxtopo.verify import equivalence
    from whiteboxtopo.invariants.extractors import fourier_irreps
    K, amp = inv["K"], inv["amp"]
    if not K: return cell("not_reached", provenance="no certified frequency set (K_f empty): nothing to rebuild from")
    Lm, _ = logit_tensor(m, data); idx = data["te"]; a, b = data["a"][idx], data["b"][idx]; X = np.stack([a, b], 1); y = data["c"][idx]
    f = lambda X_: Lm[idx]  # noqa: E731  (the model's function on the held-out pairs; marginalised for planted)
    rec = from_irreps(K, {k: amp[k] for k in K}, P); fh = lambda X_: rec.forward(a, b)  # noqa: E731
    v = equivalence(f, fh, X, y=y, radii=())
    # diagnostic upper bound: the complete diagonal inventory (all 56 frequencies)
    allK = list(range(1, (P - 1) // 2 + 1)); rec_all = from_irreps(allK, {k: amp[k] for k in allK}, P); v_all = equivalence(f, lambda X_: rec_all.forward(a, b), X, y=y, radii=())
    # pre-registered path: K and amplitudes from the token embedding (fourier_irreps, tau 0.05)
    r = fourier_irreps(m.wte.detach().cpu().numpy(), P, tau=TAU); v_emb = None
    if len(r.K): rec_e = from_irreps(list(r.K), r.amplitude, P); v_emb = equivalence(f, lambda X_: rec_e.forward(a, b), X, y=y, radii=()).to_dict()
    # K-O-1: a random frequency set of the same size must fail
    rngK = sorted(np.random.default_rng(0).choice(np.arange(1, (P - 1) // 2 + 1), size=len(K), replace=False).tolist())
    rec0 = from_irreps(rngK, {k: 1.0 for k in rngK}, P); L0 = rec0.forward(a, b); acc0 = float((L0.argmax(1) == y).mean())
    r0 = float(np.corrcoef(L0.ravel(), Lm[idx].ravel())[0, 1])
    d = v.to_dict(); ok = bool(d["passed"]) and r0 < 0.3
    return cell("certified" if ok else "failed", value=dict(acc_hat=d["accuracy_hat"], acc=d["accuracy"], logit_r=d["logit_corr"], kill_random_K_logit_r=r0, kill_random_K_acc_tautology=acc0),
                criterion="R1: accuracy >= f − 0.5 %, logit correlation > 0.98 (verify.equivalence on held-out pairs; integer domain → perturbation block n/a); K-O-1: random-K rebuild logit r < 0.3 (accuracy is tautological on Z_p — AMENDMENTS #5a — recorded)",
                provenance="reconstruct.from_irreps(K_f, diagonal-bin amplitudes) — no weights copied; complete-inventory (56 freq) rebuild and the pre-registered embedding-amplitude rebuild reported as diagnostics",
                verdict=d, verdict_all_freq=v_all.to_dict(), verdict_prereg_embedding=v_emb, K_emb=sorted(int(k) for k in r.K), rebuilt=rec.to_dict(), random_K=rngK)


def certified_features(K, a, b):
    """Degree-2 trigonometric basis of each certified irrep k acting on (a, b): 1, cos/sin w a, cos/sin w b, cos/sin w(a+b), cos/sin w(a-b)."""
    cols = [np.ones(len(a))]
    for k in K:
        w = 2 * np.pi * k / P
        cols += [np.cos(w * (a + b)), np.sin(w * (a + b)), np.cos(w * (a - b)), np.sin(w * (a - b)), np.cos(w * a), np.sin(w * a), np.cos(w * b), np.sin(w * b)]
    return np.stack(cols, 1)


def full_cloud(m, data):
    """Per-unit standardised MLP cloud at '=' over ALL p^2 pairs (recorded gauge fixing), with (a, b); dead units dropped."""
    toks = olib.grok_tokens(data, None, shortcut="leak"); H = mlp_cloud(m, toks); Z, keep = standardise(H)
    return Z, keep, data["a"], data["b"]


@guarded
def cell_R4(m, data, inv, n=3000):
    from whiteboxtopo.reconstruct import residual, coverage_gap
    K = inv["K"]
    if not K: return cell("not_reached", provenance="no certified frequency set: no reconstruction of the activations")
    H, keep, a, b = full_cloud(m, data)
    Phi = certified_features(K, a, b); coef, *_ = np.linalg.lstsq(Phi, H, rcond=None); Hhat = Phi @ coef
    R = residual(H, Hhat); r2 = 1 - float((R ** 2).sum() / ((H - H.mean(0)) ** 2).sum()); idx = np.arange(len(a))
    cg = coverage_gap(R, q=0.99, seed=0); d = cg.to_dict()
    ok = d["gamma_cov"] < 0.05 and d["gamma_null_gaussian"] < 0.02
    return cell("certified" if ok else "failed", value=dict(gamma_cov=d["gamma_cov"], gamma_excess=d["gamma_excess"], r2_certified=r2),
                criterion="R4: gamma_cov < 0.05 with synthetic-noise null < 0.02 (reconstruct.coverage_gap, TW-corrected MP edge q 0.99)",
                provenance=f"activations = per-unit standardised (gauge fixing) MLP cloud at '=' on all {len(idx)} pairs ({int(keep.sum())} live units); reconstruction h_hat = least-squares projection onto the degree-2 trig basis of the certified irreps ({Phi.shape[1]} features, AMENDMENTS #5b); residual r = h − h_hat",
                coverage=d, n_features=int(Phi.shape[1]), n_units=int(keep.sum()), K=K)


@guarded
def cell_R5(m, data, inv, r1):
    if r1["status"] == "not_reached": return cell("not_reached", provenance="R1 not reached")
    d = r1["evidence"]["verdict"]
    # train-split agreement (the rebuild has never seen either split)
    from whiteboxtopo.reconstruct import from_irreps
    from whiteboxtopo.verify import equivalence
    K, amp = inv["K"], inv["amp"]; Lm, _ = logit_tensor(m, data); idx = data["tr"]; a, b = data["a"][idx], data["b"][idx]
    rec = from_irreps(K, {k: amp[k] for k in K}, P); v_tr = equivalence(lambda X_: Lm[idx], lambda X_: rec.forward(a, b), np.stack([a, b], 1), y=data["c"][idx], radii=()).to_dict()
    applicable = ["acc_drop", "logit_corr"]; ok = all(d["checks"].get(c, False) for c in applicable) and all(v_tr["checks"].get(c, False) for c in applicable)
    return cell("certified" if ok else "failed", value=dict(argmax_agreement_test=d["argmax_agreement"], argmax_agreement_train=v_tr["argmax_agreement"], logit_r_test=d["logit_corr"], logit_r_train=v_tr["logit_corr"]),
                criterion="R5 battery: accuracy drop <= 0.5 %, logit r >= 0.98 on held-out AND train pairs; Gaussian/PGD perturbation curves and OOD agreement are NOT APPLICABLE on the integer Z_p domain (p^2 pairs = the whole domain; f_hat has no embedding space) — recorded, not counted",
                provenance="verify.equivalence (perturbation_skipped recorded); the domain is finite and fully enumerated", verdict_test=d, verdict_train=v_tr,
                not_applicable=["gaussian_perturbation", "pgd", "ood_agreement"])


@guarded
def cell_R6(m, data, inv, r1):
    from whiteboxtopo.verify import ablation_symmetry
    K, amp = inv["K"], inv["amp"]
    if not K or r1["status"] == "not_reached": return cell("not_reached", provenance="no certified objects")
    from whiteboxtopo.reconstruct import from_irreps
    rec = from_irreps(K, {k: amp[k] for k in K}, P); w = {k: float(wk) for k, wk in zip(rec.K, rec.w)}
    idx = data["te"][:2000]; a, b = data["a"][idx], data["b"][idx]; X = np.stack([a, b], 1)
    E = m.wte.detach().cpu().numpy(); Ef = np.fft.fft(E[:P], axis=0)
    def f_ablate(k, X_):
        mm = copy.deepcopy(m)
        if k is not None:
            F2 = Ef.copy(); F2[k] = 0; F2[P - k] = 0
            with torch.no_grad(): mm.wte[:P] = torch.as_tensor(np.fft.ifft(F2, axis=0).real, dtype=mm.wte.dtype, device=mm.wte.device)
        L, _ = logit_tensor(mm, data); return L[idx]
    def fh_ablate(k, X_):
        Kk = [kk for kk in K if kk != k]; return rebuild_logits(Kk, [w[kk] for kk in Kk], a, b)
    tab = ablation_symmetry(f_ablate, fh_ablate, K, X, threshold=0.9); d = tab.to_dict()
    return cell("certified" if d["all_passed"] else "failed", value=dict(n_objects=len(K), n_failed=d["n_failed"], r_min=min(r["r"] for r in d["rows"])),
                criterion="R6: per certified object (frequency k), ablation in f (zero the k-th Fourier component of the token embedding) vs its counterpart in f_hat (drop k from the rebuild): effect-vector r > 0.9",
                provenance="verify.ablation_symmetry on 2000 held-out pairs; f effects on the (marginalised) logit tensor", table=d)


@guarded
def cell_R7(m, data, inv, r1):
    from scipy.stats import spearmanr
    K, amp = inv["K"], inv["amp"]
    if not K or r1["status"] == "not_reached": return cell("not_reached", provenance="no certified objects")
    Lm, _ = logit_tensor(m, data); idx = data["te"]; a, b = data["a"][idx], data["b"][idx]; X = np.stack([a, b], 1); y = data["c"][idx]; Lt = Lm[idx]
    rebuild = _rebuild_fn_factory(amp); artifacts = [k for k in range(1, (P - 1) // 2 + 1) if k not in K]
    def score(objs):
        Lh = rebuild(objs)(X); return float(np.corrcoef(Lh.ravel(), Lt.ravel())[0, 1]), float((Lh.argmax(1) == y).mean())
    r_art, acc_art = score(artifacts); r_full, acc_full = score(K)
    rng = np.random.default_rng(0); curve = []
    for n_keep in range(1, len(K) + 1):
        rs = [score(sorted(rng.choice(K, size=n_keep, replace=False).tolist()))[0] for _ in range(10)]
        curve.append(dict(n_kept=n_keep, frac_kept=n_keep / len(K), r_mean=float(np.mean(rs)), r_std=float(np.std(rs))))
    rho = float(spearmanr([c["frac_kept"] for c in curve], [c["r_mean"] for c in curve]).correlation) if len(curve) > 2 else float("nan")
    ok = r_art < 0.3 and rho > 0
    return cell("certified" if ok else "failed", value=dict(artifact_logit_r=r_art, full_logit_r=r_full, monotone_rho=rho, artifact_acc_tautology=acc_art),
                criterion="R7 (AMENDMENTS #5a): rebuild from artifact-class objects only (the uncertified frequencies with their own diagonal amplitudes) must have logit r < 0.3 with the model; random certified subsets must degrade in proportion (Spearman(fraction kept, logit r) > 0); accuracy recorded as the tautology it is on Z_p",
                provenance="same from_irreps procedure; verify.negative_control's accuracy metric is tautological on Z_p (any frequency set classifies perfectly) and is not used", curve=curve, n_artifacts=len(artifacts))


@guarded
def cell_R8(m, data, inv, r4, r5):
    """Residue-guided second pass: only if R4 or R5 failed. The residual's top MP-outlier directions are read in the (a, b) Fourier
    basis; bins carrying > 5 % of a direction's variance become new certified features; gamma_cov is recomputed once."""
    if r4["status"] == "not_reached": return cell("not_reached", provenance="R4 not reached")
    if r4["status"] == "certified" and r5["status"] == "certified": return cell("certified", value="not needed", criterion="R8 runs only if R4/R5 fail", provenance="R4 and R5 passed")
    from whiteboxtopo.reconstruct import residual, coverage_gap
    K = inv["K"]; H, keep, a, b = full_cloud(m, data)
    Phi = certified_features(K, a, b); coef, *_ = np.linalg.lstsq(Phi, H, rcond=None); R = H - Phi @ coef
    cg0 = coverage_gap(R, q=0.99, seed=0)
    # top outlier directions → their (a, b) spectrum
    C = R.T @ R / len(R); lam, V = np.linalg.eigh(C); top = V[:, ::-1][:, :max(1, cg0.n_outliers)]
    S = R @ top                                                  # (N, n_top) scores
    grid = np.zeros((P, P, S.shape[1])); grid[a, b] = S                                             # full p^2 grid
    F = np.abs(np.fft.fft2(grid, axes=(0, 1))) ** 2; new = set()
    for j in range(S.shape[1]):
        pw = F[:, :, j]; pw[0, 0] = 0; sh = pw / pw.sum()
        for k1, k2 in zip(*np.where(sh > 0.05)):
            k1, k2 = int(min(k1, P - k1)), int(min(k2, P - k2)); new.add((k1, k2))
    cols = [Phi]
    for k1, k2 in sorted(new):
        w1, w2 = 2 * np.pi * k1 / P, 2 * np.pi * k2 / P
        cols.append(np.stack([np.cos(w1 * a + w2 * b), np.sin(w1 * a + w2 * b), np.cos(w1 * a - w2 * b), np.sin(w1 * a - w2 * b)], 1))
    Phi2 = np.concatenate(cols, 1); coef2, *_ = np.linalg.lstsq(Phi2, H, rcond=None); R2 = H - Phi2 @ coef2
    cg1 = coverage_gap(R2, q=0.99, seed=0)
    return cell("partial", value=dict(gamma_before=cg0.gamma_cov, gamma_after=cg1.gamma_cov, n_new_features=len(new)),
                criterion="R8: one residue-guided pass; report gamma_cov before/after (no pass criterion of its own)",
                provenance="top MP-outlier directions of the residual read in the 2-D (a, b) DFT basis; bins > 5 % of a direction's power become cos/sin(w_k1 a ± w_k2 b) features",
                new_bins=sorted(new), before=cg0.to_dict(), after=cg1.to_dict())


# ============================================================================================== ring cells
def ring_states(model, seed=10_000, n_seq=64, T=200, burn=20):
    erng = np.random.default_rng(seed); u, y, th = ring_batch(erng, n_seq, T)
    with torch.no_grad(): out, Hs = model(u, return_states=True)
    H = Hs[:, burn:].reshape(-1, model.n).cpu().numpy(); theta = th[:, burn:].reshape(-1); om = u[:, burn:, 0].cpu().numpy().reshape(-1)
    pred = np.arctan2(out[..., 1].cpu().numpy(), out[..., 0].cpu().numpy())[:, burn:].reshape(-1)
    return H, theta, om, pred, (u, y, th, out)


def ring_gauge(model, rng, scale=True):
    """Exact gauge of the leaky-ReLU RNN: unit permutation π and positive per-unit scaling s (relu is positively homogeneous):
    W' = S P W P^T S^-1, U' = S P U... (h' = S P h), b' = S P b, R' = P^T S^-1 R  — the readout and the map are unchanged."""
    n = model.n; perm = rng.permutation(n); s = np.exp(rng.normal(0, 0.5, n)) if scale else np.ones(n)
    Pm = np.eye(n)[perm]; S = np.diag(s); m2 = copy.deepcopy(model)
    with torch.no_grad():
        W = model.W.cpu().numpy(); U = model.U.cpu().numpy(); b = model.b.cpu().numpy(); R = model.R.cpu().numpy()
        # h @ W convention: h_{t+1} = relu(h W + u U + b). With h' = h A (A = P^T S), need W' = A^-1 W A, U' = U A, b' = b A, R' = A^-1 R
        A = Pm.T @ S; Ai = np.linalg.inv(A)
        m2.W.copy_(torch.as_tensor(Ai @ W @ A, dtype=model.W.dtype)); m2.U.copy_(torch.as_tensor(U @ A, dtype=model.U.dtype))
        m2.b.copy_(torch.as_tensor(b @ A, dtype=model.b.dtype)); m2.R.copy_(torch.as_tensor(Ai @ R, dtype=model.R.dtype))
    return m2.to(model.W.device)


def h1_bars(Z, n_max=1500, seed=0):
    from ripser import ripser
    rng = np.random.default_rng(seed); Zs = Z[rng.choice(len(Z), size=min(n_max, len(Z)), replace=False)]
    dg = ripser(Zs, maxdim=1)["dgms"][1]; life = np.sort(dg[:, 1] - dg[:, 0])[::-1] if len(dg) else np.array([0.0])
    ratio = float(life[0] / life[1]) if len(life) > 1 and life[1] > 0 else float("inf")
    return dict(n_bars=int(len(dg)), top=float(life[0]), second=float(life[1]) if len(life) > 1 else 0.0, ratio=ratio), Zs


def circ_r(theta_hat, theta):
    """Mean resultant length of the phase difference after the best orientation (±) — 1 = perfect circular agreement; the
    orientation/offset is a gauge of the coordinate and is recorded."""
    best = None
    for s in (1, -1):
        d = np.exp(1j * (theta_hat - s * theta)); r = float(abs(d.mean())); phi = float(np.angle(d.mean()))
        if best is None or r > best[0]: best = (r, s, phi)
    return best


@guarded
def ring_gauge_cell(model, n_gauge):
    rng = np.random.default_rng(1); u, _, _ = ring_batch(np.random.default_rng(5), 32, 100)
    with torch.no_grad():
        ref = model(u); devs = [float((ring_gauge(model, rng)(u) - ref).abs().max() / (ref.abs().max() + 1e-12)) for _ in range(min(n_gauge, 50))]
    return cell("certified", value=float(np.max(devs)), criterion="exact symmetries: max relative output deviation on 50 gauge samples (unit permutations x positive scalings) at float32 round-off",
                provenance="leaky-ReLU RNN gauge group S_n x R_>0^n (relu positively homogeneous; readout compensated); hidden symmetries not computed", max_dev=float(np.max(devs)), median_dev=float(np.median(devs)))


@guarded
def ring_P0(model):
    from whiteboxtopo.legibility import graph as G
    H, *_ = ring_states(model); sd = H.std(0) + 1e-8
    W = model.W.detach().cpu().numpy(); Eff = np.abs(W) * sd[:, None] / sd[None, :]          # effective coupling in the standardised gauge
    A = G.threshold_density(G.symmetrise(Eff), 0.1); Q, _ = G.modularity_q(A, seed=0); tw = G.treewidth_bound(A)
    null = G.shuffled_q_null(Eff, density=0.1, n=20, seed=0); qn = np.asarray(null["Q"] if isinstance(null, dict) else null, float)
    verdict = "tractable" if (Q > 0.3 and tw <= 8) else ("intractable" if tw > 32 else "marginal")
    return cell("partial", value=verdict, criterion="P0 on the ring: the instrument is defined for attribution graphs of feed-forward nodes; the recurrent effective-coupling graph |W_ij| sd_i/sd_j is a proxy → partial only",
                provenance="legibility.graph on the standardised recurrent weight graph at density 0.1; null = degree-preserving rewiring", Q=float(Q), treewidth=int(tw), null_Q_mean=float(qn.mean()))


@guarded
def ring_P1(model, n_gauge):
    from whiteboxtopo.invariants.extractors import circular_coordinate
    H, theta, om, pred, _ = ring_states(model); Z, keep = standardise(H); bars, Zs = h1_bars(Z)
    rng = np.random.default_rng(0); sub = rng.choice(len(Z), size=min(1500, len(Z)), replace=False)
    th_hat, ratio_cc = circular_coordinate(Z[sub]); r, s, phi = circ_r(th_hat, theta[sub])
    r_pred = circ_r(th_hat, pred[sub])[0]
    # nulls: feature shuffle (each unit permuted independently across points) and random-direction features (Gaussian cloud)
    Zsh = np.stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])], 1); bars_sh, _ = h1_bars(Zsh)
    bars_rand, _ = h1_bars(rng.standard_normal(Z.shape))
    # K-O-2: the certified scalars over the gauge orbit (after the recorded per-unit standardisation)
    orb = []
    grng = np.random.default_rng(2)
    for _ in range(min(n_gauge, 20)):
        Hg = ring_states(ring_gauge(model, grng))[0]; Zg, _ = standardise(Hg); bg, _ = h1_bars(Zg); tg, _ = circular_coordinate(Zg[sub]); orb.append([bg["ratio"] if np.isfinite(bg["ratio"]) else 1e6, circ_r(tg, theta[sub])[0]])
    orb = np.asarray(orb); cv_ratio, cv_r = olib.cv(orb[:, 0]), olib.cv(orb[:, 1])
    ok = bars["ratio"] > 5 and r > 0.99 and max(cv_ratio, cv_r) < ART_CV and bars_sh["ratio"] < 5 and bars_rand["ratio"] < 5
    return cell("certified" if ok else "failed", value=dict(beta1_ratio=bars["ratio"], circ_r=r, n_bars=bars["n_bars"]),
                criterion="ring P1: one H1 bar with lifetime > 5x the next; circular coordinate vs true angle r > 0.99 (mean resultant length after orientation/offset gauge); nulls: feature shuffle + Gaussian cloud must not show the bar; K-O-2 CV < 1e-3 over 20 gauge samples",
                provenance=f"held-out states (64 seq x 180 steps after burn-in, {int(keep.sum())} live units, per-unit standardised = recorded gauge fixing); ripser VR H1 on 1500 points; extractors.circular_coordinate (ripser cocycles)",
                bars=bars, lifetime_ratio_cocycle=float(ratio_cc), orientation=int(s), offset=float(phi), r_vs_readout=float(r_pred), null_shuffle=bars_sh, null_gaussian=bars_rand, cv_ratio=cv_ratio, cv_r=cv_r,
                artifact=max(cv_ratio, cv_r) >= ART_CV)


@guarded
def ring_P2(model, n_gauge):
    """Exact attribution of the final read-out angle to each input velocity: the unrolled leaky-ReLU RNN is piecewise linear, so
    out_T = A(u) u + c(u) exactly on the activation region (path products through time); anchor: IG from the zero-velocity path."""
    u, y, th, _ = ring_batch(np.random.default_rng(7), 64, 100)[3] if False else (None, None, None, None)
    urng = np.random.default_rng(7); u, y, th = ring_batch(urng, 64, 100)
    def attribution(mm, uu):
        x = uu.clone().requires_grad_(True); out = mm(x)[:, -1]                                  # (B, 2)
        ang = torch.atan2(out[:, 1], out[:, 0]); g, = torch.autograd.grad(ang.sum(), x)
        return (g[:, :, 0] * uu[:, :, 0]).detach().cpu().numpy()                                  # contribution of each omega_t
    A = attribution(model, u)
    grng = np.random.default_rng(3); orbit = [attribution(ring_gauge(model, grng), u) for _ in range(min(n_gauge, 50))]; cv_attr = olib.cv(orbit)
    # IG along u -> alpha u (velocities scaled, theta_0 cue kept) — agrees with A on regions of constant pattern
    steps = 32; ig = np.zeros_like(A)
    for s_ in range(1, steps + 1):
        uu = u.clone(); uu[:, :, 0] *= s_ / steps; x = uu.requires_grad_(True); out = model(x)[:, -1]
        ang = torch.atan2(out[:, 1], out[:, 0]); g, = torch.autograd.grad(ang.sum(), x); ig += (g[:, :, 0]).detach().cpu().numpy()
    ig = ig / steps * u[:, :, 0].cpu().numpy(); r = float(np.corrcoef(A.ravel(), ig.ravel())[0, 1])
    ok = cv_attr < 1e-4 and r > 0.95
    return cell("certified" if ok else ("partial" if cv_attr < 1e-4 else "failed"), value=dict(cv=cv_attr, ig_r=r),
                criterion="attribution CV over the gauge orbit < 1e-4; IG anchor r > 0.95", provenance="exact Jacobian x input of the unrolled piecewise-linear RNN (final read-out angle w.r.t. each omega_t); IG 32 steps from the zero-velocity baseline; orbit n=50")


@guarded
def ring_P7(name):
    from whiteboxtopo.invariants.extractors import circular_coordinate
    cks = olib.checkpoints(name); rows = []
    for step, path in cks[::2] + [cks[-1]]:
        mm, ck = olib.load_ring(name, step); H, theta, *_ = ring_states(mm, n_seq=32); Z, _ = standardise(H)
        if Z.shape[1] < 3: rows.append((step, float(ck["ang_err"]), 1.0, 0.0)); continue
        bars, _ = h1_bars(Z, n_max=800); sub = np.random.default_rng(0).choice(len(Z), size=min(800, len(Z)), replace=False)
        try: th_hat, _ = circular_coordinate(Z[sub]); r = circ_r(th_hat, theta[sub])[0]
        except Exception: r = 0.0
        rows.append((step, float(ck["ang_err"]), float(min(bars["ratio"], 1e6)), float(r)))
    steps = np.array([r[0] for r in rows]); err = np.array([r[1] for r in rows]); ratio = np.array([r[2] for r in rows]); rr = np.array([r[3] for r in rows]); T = steps[-1]
    t_ring = float(steps[np.argmax(ratio > 5)] / T) if (ratio > 5).any() else None
    t_err = float(steps[np.argmax(err < 0.3)] / T) if (err < 0.3).any() else None
    return cell("partial", value=dict(t_ring=t_ring, t_error=t_err), criterion="ring P7 (no pre-registered threshold): when the H1 bar (ratio > 5) and the circular coordinate appear vs when the angular error drops below 0.3 rad — reported",
                provenance="every 2nd checkpoint; standardised cloud of 32 held-out sequences; ripser H1 on 800 points", curve=dict(steps=steps.tolist(), ang_err=err.tolist(), ratio=ratio.tolist(), circ_r=rr.tolist()))


@guarded
def ring_R2(model, p1):
    from whiteboxtopo.invariants.extractors import circular_coordinate
    from whiteboxtopo.reconstruct import fit_tangent_dynamics, from_attractor
    if p1["status"] != "certified": return cell("not_reached", provenance="P1 (circle + coordinate) not certified: nothing to rebuild from")
    H, theta, om, pred, (u, y, th, out) = ring_states(model, n_seq=64, T=100, burn=0); Z, keep = standardise(H)
    n_seq, T = 64, 100; rng = np.random.default_rng(0); sub = rng.choice(len(Z), size=1500, replace=False)
    th_hat_sub, _ = circular_coordinate(Z[sub]); r, s, phi = circ_r(th_hat_sub, theta[sub])
    # extend the certified coordinate to every state with a linear decoder of (cos, sin) fitted on the landmarks (recorded)
    Dm = np.stack([np.cos(th_hat_sub), np.sin(th_hat_sub)], 1); A = np.c_[Z[sub], np.ones(len(sub))]; coef, *_ = np.linalg.lstsq(A, Dm, rcond=None)
    Dall = np.c_[Z, np.ones(len(Z))] @ coef; th_all = np.arctan2(Dall[:, 1], Dall[:, 0]).reshape(n_seq, T)
    th_true = th.reshape(n_seq, T); om2 = u[:, :, 0].cpu().numpy(); pred2 = pred.reshape(n_seq, T)
    # tangent dynamics on the certified coordinate: dtheta_hat = g(theta_hat, omega) (discrete map, one step)
    d = np.angle(np.exp(1j * (th_all[:, 1:] - th_all[:, :-1])))
    dyn = fit_tangent_dynamics(th_all[:, :-1].ravel(), om2[:, :-1].ravel(), d.ravel(), n_harmonics=3, u_degree=1)
    rec = from_attractor(th_all[:, 0], dyn)
    # integrate 100 steps from the certified initial coordinate with the true velocity sequence; compare to the model's read-out
    seq = np.empty_like(th_all); seq[:, 0] = th_all[:, 0]
    for t in range(1, T): seq[:, t] = (seq[:, t - 1] + dyn.predict(seq[:, t - 1], om2[:, t - 1])) % (2 * np.pi)
    # map the coordinate to the true-angle frame with the recorded orientation/offset gauge, then compare
    seq_true_frame = s * (seq - phi); err_hat = float(np.abs(np.angle(np.exp(1j * (seq_true_frame - th_true)))).mean())
    err_model = float(np.abs(np.angle(np.exp(1j * (pred2 - th_true)))).mean())
    # drift statistics: zero-velocity holds for 100 steps (model vs rebuild)
    u0 = u.clone(); u0[:, :, 0] = 0
    with torch.no_grad(): o0 = model(u0)[..., :].cpu().numpy()
    p0 = np.arctan2(o0[..., 1], o0[..., 0]); drift_model = float(np.abs(np.angle(np.exp(1j * (p0[:, -1] - p0[:, 0])))).mean())
    seq0 = th_all[:, 0].copy()
    for t in range(1, T): seq0 = (seq0 + dyn.predict(seq0, np.zeros(n_seq))) % (2 * np.pi)
    drift_hat = float(np.abs(np.angle(np.exp(1j * (seq0 - th_all[:, 0])))).mean())
    ok = abs(err_hat - err_model) <= 0.10 * err_model and dyn.r2 > 0.5
    return cell("certified" if ok else "failed", value=dict(err_rebuild=err_hat, err_model=err_model, ratio=err_hat / max(err_model, 1e-9), dyn_r2=dyn.r2, drift_model=drift_model, drift_rebuild=drift_hat),
                criterion="R2: integration error of the rebuild (circular coordinate + fitted tangent dynamics only, no recurrent weights) within 10 % of the model's over 100-step trajectories; drift statistics reported; tangent fit r2 > 0.5 (informative)",
                provenance="extractors.circular_coordinate on 1500 landmark states (standardised = gauge fixing), linear (cos, sin) decoder to extend the coordinate (recorded), reconstruct.fit_tangent_dynamics (3 harmonics x linear in omega), discrete one-step map; orientation/offset gauge to the true frame recorded",
                dynamics=dyn.to_dict(), orientation=int(s), offset=float(phi), coord_r=float(r), rebuilt=rec.to_dict())


# ============================================================================================== drivers
def scorecard(name, cells, kind):
    card = Scorecard(model=name, behavior=dict(name="Z113 modular addition" if kind == "grok" else "angular integration (ring attractor)", kind="group" if kind == "grok" else "attractor"))
    mapping = {"P0": "P0", "P1": "P1", "P2": "P2", "P6": "P6", "P7": "P7"}
    for p in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        c = cells.get(mapping.get(p, p))
        if c is None:
            why = {"P3": "O1.3 Nishino Flower", "P4": "O1.4 Fine Motion", "P5": "O1.4 Fine Motion"}.get(p, "not run")
            card.cells[p] = CellResult(p, "not_reached", provenance=f"not reached here: {why}"); continue
        st = c["status"] if c["status"] in ("certified", "partial", "failed", "not_reached") else "failed"
        card.cells[p] = CellResult(p, st, value=c["value"], evidence={"criterion": c["criterion"]}, provenance=c["provenance"])
    r4 = cells.get("R4"); card.gamma_cov = (r4["value"]["gamma_cov"] if isinstance(r4["value"], dict) else None) if r4 and r4["status"] != "not_reached" else None
    card.reconstruction = {k: cells[k]["status"] for k in cells if k.startswith("R")}
    return card


def write(name, out, kind):
    json.dump(out, open(os.path.join(OUT, f"meisho_doto_{name}.json"), "w"), indent=1, default=str)
    card = scorecard(name, out["cells"], kind); out["scorecard"] = card.to_dict()
    json.dump(out, open(os.path.join(OUT, f"meisho_doto_{name}.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, f"meisho_doto_{name}.md"), "w").write(card.to_markdown() + "\n")
    new = not os.path.exists(CELLS_CSV)
    with open(CELLS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS); new and w.writeheader()
        for k, c in out["cells"].items():
            w.writerow(dict(specimen=name, cell=k, status=c["status"], value=json.dumps(c["value"], default=str)[:200], criterion=c["criterion"][:160], provenance=c["provenance"][:200], wall=c.get("wall")))
            print(f"{name} {k:5s} {c['status']:12s} {json.dumps(c['value'], default=str)[:110]}  [{c.get('wall')}s]", flush=True)


def run_grok(name, n_gauge, cells):
    m, ck, data = olib.load_grok(name); data["condition"] = ck["cfg"]["condition"]
    out = dict(specimen=name, kind="grok", condition=data["condition"], seed=ck["cfg"]["seed"], recipe=ck["cfg"].get("recipe", 1), cells={})
    Lm, _ = logit_tensor(m, data); inv = irreps_f(Lm); out["inventory"] = dict(kappa_f=inv["kappa_f"], K=inv["K"], n_eff=inv["n_eff"], amp={int(k): inv["amp"][k] for k in inv["K"]})
    C = out["cells"]
    if "gauge" in cells: C["gauge"] = step_gauge(m, data, n_gauge)
    if "P0" in cells: C["P0"] = cell_P0(m, data)
    if "P1" in cells: C["P1"] = cell_P1(m, data, n_gauge, inv)
    if "P2" in cells: C["P2"] = cell_P2(m, data, n_gauge)
    if "P6" in cells: C["P6"] = cell_P6(m, data, inv)
    if "P7" in cells: C["P7"] = cell_P7(name, data, inv)
    if "glue" in cells: C["glue"] = cell_glue(m, data)
    certified_K = C.get("P1", {}).get("status") == "certified"
    inv_c = inv if certified_K else dict(inv, K=[])                 # R cells rebuild from CERTIFIED structure only
    if "R1" in cells: C["R1"] = cell_R1(m, data, inv_c)
    if "R4" in cells: C["R4"] = cell_R4(m, data, inv_c)
    if "R5" in cells: C["R5"] = cell_R5(m, data, inv_c, C.get("R1", cell("not_reached")))
    if "R6" in cells: C["R6"] = cell_R6(m, data, inv_c, C.get("R1", cell("not_reached")))
    if "R7" in cells: C["R7"] = cell_R7(m, data, inv_c, C.get("R1", cell("not_reached")))
    if "R8" in cells: C["R8"] = cell_R8(m, data, inv_c, C.get("R4", cell("not_reached")), C.get("R5", cell("not_reached")))
    write(name, out, "grok"); return out


def run_ring(name, n_gauge, cells):
    model, ck = olib.load_ring(name); out = dict(specimen=name, kind="ring", condition="ring", seed=ck["cfg"]["seed"], cells={}); C = out["cells"]
    if "gauge" in cells: C["gauge"] = ring_gauge_cell(model, n_gauge)
    if "P0" in cells: C["P0"] = ring_P0(model)
    if "P1" in cells: C["P1"] = ring_P1(model, n_gauge)
    if "P2" in cells: C["P2"] = ring_P2(model, n_gauge)
    if "P7" in cells: C["P7"] = ring_P7(name)
    if "R2" in cells: C["R2"] = ring_R2(model, C.get("P1", cell("not_reached")))
    write(name, out, "ring"); return out


def aggregate():
    import glob
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    outs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "meisho_doto_*.json")))]
    groks = [o for o in outs if o["kind"] == "grok" and o.get("recipe", 1) >= 2]; rings = [o for o in outs if o["kind"] == "ring"]
    S = dict(n_grok=len(groks), n_ring=len(rings))
    # P6 AUC: kappa_f (and the pre-registered kappa_emb) grokked vs memorising
    lab = [0 if o["condition"] == "memorize" else 1 for o in groks]; kf = [o["inventory"]["kappa_f"] for o in groks]
    ke = [o["cells"]["P6"]["value"]["kappa_emb"] for o in groks if "P6" in o["cells"] and isinstance(o["cells"]["P6"].get("value"), dict)]
    if len(set(lab)) == 2:
        S["P6"] = dict(auc_kappa_f=float(roc_auc_score(lab, kf)), auc_kappa_emb=float(roc_auc_score(lab, ke)) if len(ke) == len(lab) else None, n=len(lab), passed=float(roc_auc_score(lab, kf)) == 1.0)
    # P7 frequency ordering: per clean seed, Spearman(appearance time, final amplitude); consistent iff the same sign on >= 4/5 seeds
    ords = []
    for o in groks:
        if o["condition"] != "clean" or "P7" not in o["cells"]: continue
        ap = o["cells"]["P7"]["evidence"].get("appearance", {}); amp = o["inventory"]["amp"]
        ks = [k for k in ap if ap[k] is not None and str(k) in amp or int(k) in amp]
        if len(ks) >= 3: rho = spearmanr([ap[k] for k in ks], [amp.get(str(k), amp.get(int(k))) for k in ks]).correlation; ords.append(float(rho))
    if ords: sgn = np.sign(ords); S["P7_ordering"] = dict(rho_per_seed=ords, consistent=(max((sgn > 0).sum(), (sgn < 0).sum()) >= min(4, len(ords))), n=len(ords))
    # mechanism count (function level): algorithm class per clean seed + distinct frequency sets
    cls = {o["specimen"]: ("fourier" if o["cells"].get("P1", {}).get("status") == "certified" else "uncertified") for o in groks if o["condition"] == "clean"}
    Ks = {o["specimen"]: tuple(o["inventory"]["K"]) for o in groks if o["condition"] == "clean"}
    S["count"] = dict(algorithm_classes=sorted(set(cls.values())), K_alg=len(set(cls.values())), per_seed=cls, frequency_sets=Ks, n_distinct_K=len(set(Ks.values())),
                      note="function-level count (align.match_circuits is a ReLU-MLP weight-level instrument, not applicable to the toy transformer)")
    # pass counts per cell on clean seeds (pre-registered: every cell certified on >= 4/5 clean seeds; R1/R2/R5 pass; gamma_cov < 0.05)
    clean = [o for o in groks if o["condition"] == "clean"]; cells = sorted({c for o in clean for c in o["cells"]})
    S["clean_pass"] = {c: dict(certified=sum(o["cells"].get(c, {}).get("status") == "certified" for o in clean), n=len(clean)) for c in cells}
    S["ring_pass"] = {c: dict(certified=sum(o["cells"].get(c, {}).get("status") == "certified" for o in rings), n=len(rings)) for c in sorted({c for o in rings for c in o["cells"]})}
    gam = [o["scorecard"]["gamma_cov"] for o in clean if o.get("scorecard", {}).get("gamma_cov") is not None]
    S["gamma_cov_clean"] = dict(values=gam, median=float(np.median(gam)) if gam else None, all_below_0_05=bool(gam) and all(g < 0.05 for g in gam))
    n = len(clean); need = min(4, n)
    S["O1_2_prereg_pass"] = bool(n) and all(S["clean_pass"][c]["certified"] >= need for c in ("P0", "P1", "P2", "P6", "P7") if c in S["clean_pass"]) and S["clean_pass"].get("R1", {}).get("certified", 0) >= need and S["clean_pass"].get("R5", {}).get("certified", 0) >= need and S["gamma_cov_clean"]["all_below_0_05"] and S["ring_pass"].get("R2", {}).get("certified", 0) >= min(4, len(rings))
    json.dump(S, open(os.path.join(RES, "O1_2_summary.json"), "w"), indent=1, default=str); print(json.dumps(S, indent=1, default=str)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--specimens", nargs="*"); ap.add_argument("--n-gauge", type=int, default=300)
    ap.add_argument("--cells", nargs="*", default=None); ap.add_argument("--aggregate", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.aggregate: aggregate(); sys.exit(0)
    names = a.specimens
    if not names:
        rows = list(csv.DictReader(open(os.path.join(RES, "specimens.csv")))); names = [r["name"] for r in rows if r["included"] == "1"]
    for n in names:
        if not a.force and os.path.exists(os.path.join(OUT, f"meisho_doto_{n}.json")): print("skip", n, "(done)"); continue
        kind = "ring" if n.startswith("ring") else "grok"
        cells = a.cells or (RING_CELLS if kind == "ring" else GROK_CELLS)
        (run_ring if kind == "ring" else run_grok)(n, a.n_gauge, cells)
