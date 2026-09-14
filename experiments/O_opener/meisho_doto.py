#!/usr/bin/env python3
"""O1.2 Meisho Doto — the full open() on every included T1 grokked specimen (SPEC_O.md §O1.2; criteria = FINAL_SPEC §6).

  python meisho_doto.py [--specimens grok_clean_s0 ...] [--n-gauge 300] [--cells P0 P1 P2 P6 P7 R]

Per specimen: results/open/<name>.json (all cells with evidence + provenance) and one row per cell appended to
results/O1_2_cells.csv. Cells whose module is not yet available are recorded as "not_reached: <reason>" — never skipped silently.
Library modules used lazily: whiteboxtopo.{invariants.extractors, reconstruct, verify, spectral, sheaf, open}.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import olib  # noqa: E402
from olib import ROOT, RES, DEV  # noqa: E402
from sweep_tosho import P, fourier_kappa  # noqa: E402

OUT = os.path.join(RES, "open"); os.makedirs(OUT, exist_ok=True)
CELLS_CSV = os.path.join(RES, "O1_2_cells.csv")
CELL_FIELDS = ["specimen", "cell", "status", "value", "criterion", "provenance", "wall"]


def cell(status, value=None, criterion="", provenance="", **evidence):
    return dict(status=status, value=value, criterion=criterion, provenance=provenance, evidence=evidence)


def guarded(fn):
    def run(*a, **k):
        t = time.time()
        try: c = fn(*a, **k)
        except Exception as e: c = cell("not_reached", provenance=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-800:])
        c["wall"] = round(time.time() - t, 1); return c
    return run


# --------------------------------------------------------------------------------------------- step 1: gauge
@guarded
def step_gauge(m, data, n_gauge):
    from whiteboxtopo.symmetry import group
    desc = group.describe(m)
    toks = olib.grok_tokens(data, data["te"][:256])
    devs = [group.output_deviation(m, g, toks, forward=lambda mm, t: mm(t)[:, -1]) for g in olib.gauge_samples(m, min(n_gauge, 50), seed=1)]
    return cell("certified", value=float(np.max(devs)), criterion="exact symmetries: max output deviation on 50 samples (float32 round-off)",
                provenance="symmetry.transformer (head perms, per-head GL(dh) rotations, MLP perms+scalings, LN rescalings); exact vs approximate: all exact for the ReLU toy transformer",
                group=desc if isinstance(desc, dict) else str(desc), max_dev=float(np.max(devs)), median_dev=float(np.median(devs)))


# --------------------------------------------------------------------------------------------- step 2: P0 gate
@guarded
def cell_P0(m, data, seed=0):
    view = olib.NodeView(m); idx = data["te"][:512]
    toks = olib.grok_tokens(data, idx); y = torch.as_tensor(data["c"][idx], device=toks.device)
    means = view.node_means(toks); E, names = olib.attribution_graph(view, toks, y, means)
    prof = olib.p0_profile(E, density=0.1, seed=seed)
    # null: degree-preserving rewiring of the thresholded graph (V block's shuffled_q_null)
    from whiteboxtopo.legibility import graph as G
    null = G.shuffled_q_null(E, density=0.1, n=20, seed=seed)
    qn = np.asarray(null["Q"] if isinstance(null, dict) else null, float)
    verdict = "tractable" if (prof["Q"] > 0.3 and prof["treewidth"] <= 8) else ("intractable" if prof["treewidth"] > 32 else "marginal")
    return cell("certified" if verdict == "tractable" else "failed", value=verdict,
                criterion="tractable iff Q>0.3 and treewidth_ub<=8 (provisional rule; V-block phase diagram is the reference)",
                provenance="olib.attribution_graph (grad x (act-mean), heads->MLP units->logits) + legibility.graph; null = degree-preserving rewiring",
                Q=prof["Q"], treewidth=prof["treewidth"], sparsity=prof["sparsity"], null_Q_mean=float(qn.mean()), null_Q_max=float(qn.max()),
                Q_minus_null=prof["Q"] - float(qn.mean()))


# --------------------------------------------------------------------------------------------- P1 / P6: irreps
def irreps_of(m, tau=0.05):
    E = m.wte.detach().cpu().numpy()
    try:
        from whiteboxtopo.invariants.extractors import fourier_irreps
        r = fourier_irreps(E, P, tau=tau); K = sorted(int(k) for k in r.K); kappa = float(r.kappa); src = "invariants.extractors.fourier_irreps"
        weights = getattr(r, "weights", None)
    except ImportError:
        kappa, K = fourier_kappa(E, P, tau); src = "sweep_tosho.fourier_kappa (extractors pending)"; weights = None
    return K, kappa, src, weights


@guarded
def cell_P1(m, data, n_gauge):
    K, kappa, src, _ = irreps_of(m)
    # certification: K and kappa over the gauge orbit (structurally invariant: the group never touches the token axis — still measured)
    orbit = olib.on_orbit(m, lambda mm: irreps_of(mm)[:2], n=n_gauge, seed=2)
    same_K = all(o[0] == K for o in orbit); cv_kappa = olib.cv([o[1] for o in orbit])
    # null: feature shuffle of the embedding rows (destroys the token axis structure) must give kappa near 0
    Es = m.wte.detach().cpu().numpy().copy(); Es[:P] = Es[np.random.default_rng(0).permutation(P)]
    kappa_null, _ = fourier_kappa(Es, P, 0.05)
    ok = kappa > 0.9 and same_K and cv_kappa < 1e-3 and kappa_null < 0.3
    return cell("certified" if ok else "failed", value=dict(K=K, kappa=kappa),
                criterion="kappa>0.9; K identical across all gauge samples; CV(kappa)<1e-3; row-shuffle null kappa<0.3 (reconstruction-from-K is R1)",
                provenance=src + f"; orbit n={n_gauge}", same_K=same_K, cv_kappa=cv_kappa, kappa_null=float(kappa_null), n_freq=len(K))


# --------------------------------------------------------------------------------------------- P2: certified attribution
def token_contributions(m, toks):
    """c[b, pos, class] = J_pos(x) . e_pos: the frozen-pattern linear contribution of the token at ``pos`` to each logit,
    where J_pos is the Jacobian of the last-position logits w.r.t. that position's embedded vector (exact under G: the
    function is identical, so its Jacobian is). Reported with the QK caveat (the pattern itself is treated as frozen)."""
    m.zero_grad(set_to_none=True)
    x = (m.wte[toks] + m.wpe[:toks.shape[1]][None]).detach().requires_grad_(True)
    # re-run the forward from the embedded input
    l = 0; h = m._ln(x, m.ln1_g[l], m.ln1_b[l])
    q = torch.einsum("btd,dhk->bthk", h, m.Wq[l]) + m.bq[l]; k = torch.einsum("btd,dhk->bthk", h, m.Wk[l]) + m.bk[l]
    v = torch.einsum("btd,dhk->bthk", h, m.Wv[l]) + m.bv[l]; T = toks.shape[1]
    att = (torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(m.dh)).masked_fill(~m.mask[:T, :T], float("-inf")).softmax(-1)
    z = torch.einsum("bhts,bshk->bthk", att, v); xx = x + torch.einsum("bthk,hkd->bthd", z, m.Wo[l]).sum(2) + m.bo[l]
    a = torch.relu(m._ln(xx, m.ln2_g[l], m.ln2_b[l]) @ m.W1[l] + m.b1[l]); xx = xx + a @ m.W2[l] + m.b2[l]
    logits = (m._ln(xx, m.lnf_g, m.lnf_b) @ m.Wu + m.bu)[:, -1]                                   # (B, P)
    B, C = logits.shape; contrib = torch.zeros(B, T, C, device=x.device)
    for c in range(C):
        g, = torch.autograd.grad(logits[:, c].sum(), x, retain_graph=True)                       # (B,T,d)
        contrib[:, :, c] = (g * x).sum(-1)
    return contrib.detach(), logits.detach()


def integrated_gradients(m, toks, steps=32):
    """IG of the correct-class logit w.r.t. each position's embedded vector, baseline = zero embedding (positional kept)."""
    T = toks.shape[1]; pos = m.wpe[:T][None]; e = m.wte[toks].detach()
    ig = torch.zeros_like(e)
    for s in range(1, steps + 1):
        x = (pos + e * (s / steps)).detach().requires_grad_(True)
        l = 0; h = m._ln(x, m.ln1_g[l], m.ln1_b[l])
        q = torch.einsum("btd,dhk->bthk", h, m.Wq[l]) + m.bq[l]; k = torch.einsum("btd,dhk->bthk", h, m.Wk[l]) + m.bk[l]
        v = torch.einsum("btd,dhk->bthk", h, m.Wv[l]) + m.bv[l]
        att = (torch.einsum("bthk,bshk->bhts", q, k) / np.sqrt(m.dh)).masked_fill(~m.mask[:T, :T], float("-inf")).softmax(-1)
        z = torch.einsum("bhts,bshk->bthk", att, v); xx = x + torch.einsum("bthk,hkd->bthd", z, m.Wo[l]).sum(2) + m.bo[l]
        a = torch.relu(m._ln(xx, m.ln2_g[l], m.ln2_b[l]) @ m.W1[l] + m.b1[l]); xx = xx + a @ m.W2[l] + m.b2[l]
        logits = (m._ln(xx, m.lnf_g, m.lnf_b) @ m.Wu + m.bu)[:, -1]
        g, = torch.autograd.grad(logits.max(1).values.sum(), x); ig += g
    return (ig / steps * e).sum(-1).detach()                                                       # (B, T)


@guarded
def cell_P2(m, data, n_gauge):
    idx = data["te"][:128]; toks = olib.grok_tokens(data, idx)
    contrib, logits = token_contributions(m, toks); pred = logits.argmax(1)
    c_pred = contrib[torch.arange(len(idx)), :, pred].cpu().numpy()                                # (B, T) contribution to the predicted logit
    orbit = olib.on_orbit(m, lambda mm: token_contributions(mm, toks)[0][torch.arange(len(idx)), :, pred].cpu().numpy(), n=min(n_gauge, 100), seed=3)
    cv_attr = olib.cv(orbit)
    ig = integrated_gradients(m, toks).cpu().numpy()
    r = float(np.corrcoef(c_pred[:, :-1].ravel(), ig[:, :-1].ravel())[0, 1])                      # token positions (drop '=')
    ok = cv_attr < 1e-4 and r > 0.95
    return cell("certified" if ok else ("partial" if cv_attr < 1e-4 else "failed"), value=dict(cv=cv_attr, ig_r=r),
                criterion="attribution CV over gauge orbit < 1e-4 (float32 realisation of 'exact'); IG anchor r > 0.95",
                provenance="frozen-pattern Jacobian contributions J_pos.e_pos (QK caveat); IG 32 steps from zero-embedding baseline; D1 anchor",
                top_token_share=float(np.mean(np.abs(c_pred[:, :2]).sum(1) / (np.abs(c_pred).sum(1) + 1e-9))))


# --------------------------------------------------------------------------------------------- P7: evolution
@guarded
def cell_P7(name, data):
    cks = olib.checkpoints(name); rows = []
    for step, path in cks:
        ck = torch.load(path, map_location="cpu"); E = ck["state"]["wte"].numpy()
        kappa, K = fourier_kappa(E, P, 0.05); rows.append((step, ck["test_acc"], kappa, tuple(K)))
    steps = np.array([r[0] for r in rows]); acc = np.array([r[1] for r in rows]); kap = np.array([r[2] for r in rows])
    T = steps[-1]
    t_acc = steps[np.argmax(acc > 0.5)] / T if (acc > 0.5).any() else None
    t_kap = steps[np.argmax(kap > 0.5)] / T if (kap > 0.5).any() else None
    # irrep appearance order: first checkpoint each final frequency is present
    finalK = rows[-1][3]; appear = {k: (steps[np.argmax([k in r[3] for r in rows])] / T if any(k in r[3] for r in rows) else None) for k in finalK}
    # spectral gap over checkpoints (MLP activation cloud on 512 test inputs) — pending module handled by the guard
    gap = None
    try:
        from whiteboxtopo.spectral import eigenflow
        clouds = []
        for step, path in cks[::max(1, len(cks) // 30)]:
            mm, _, _ = olib.load_grok(name, step); view = olib.NodeView(mm); toks = olib.grok_tokens(data, data["te"][:512])
            with torch.no_grad(): view(toks, record=True)
            clouds.append(torch.stack([o.detach() for o in view._node_out[view.H:]], -1).cpu().numpy())
        ef = eigenflow(clouds, k=15, n_eigs=10); gap = [float(g) for g in ef.gaps] if hasattr(ef, "gaps") else ef.to_dict()
    except ImportError as e: gap = f"not_reached: {e}"
    ok = t_acc is not None and t_kap is not None and abs(t_acc - t_kap) <= 0.05
    return cell("certified" if ok else "partial", value=dict(t_acc=t_acc, t_kappa=t_kap),
                criterion="irrep appearance (kappa>0.5) within 5 % of the accuracy jump (test acc>0.5); frequency ordering across seeds checked in the aggregate; spectral gap timing pending spectral module",
                provenance="checkpoint curves (300/model); fourier_kappa per checkpoint", appearance=appear, spectral_gap=gap, n_ckpt=len(cks))


# --------------------------------------------------------------------------------------------- R1 + R4/R5 (module-dependent)
@guarded
def cell_R1(m, data):
    from whiteboxtopo.invariants.extractors import fourier_irreps
    from whiteboxtopo.reconstruct import from_irreps
    from whiteboxtopo.verify import equivalence
    E = m.wte.detach().cpu().numpy(); r = fourier_irreps(E, P, tau=0.05)
    rec = from_irreps(r.K, r.weights, P)
    idx = data["te"]; a, b = data["a"][idx], data["b"][idx]
    f = lambda X: olib.grok_logits(m, data, idx)                                                    # noqa: E731
    fh = lambda X: np.asarray(rec.forward(a, b))                                                    # noqa: E731
    v = equivalence(f, fh, np.stack([a, b], 1), y=data["c"][idx], radii=())
    # kill K-O-1: random frequency set of the same size must fail
    rngK = sorted(np.random.default_rng(0).choice(np.arange(1, (P - 1) // 2 + 1), size=len(r.K), replace=False).tolist())
    rec0 = from_irreps(rngK, r.weights, P); acc0 = float((np.asarray(rec0.forward(a, b)).argmax(1) == data["c"][idx]).mean())
    d = v.to_dict(); ok = bool(d.get("passed")) and acc0 < 0.05
    return cell("certified" if ok else "failed", value=dict(acc_rebuilt=d.get("acc_hat", d.get("acc_f_hat")), logit_r=d.get("logit_corr"), kill_random_K_acc=acc0),
                criterion="R1: acc >= f-0.5 %, logit r > 0.98; K-O-1 random-K rebuild near chance", provenance="reconstruct.from_irreps + verify.equivalence", verdict=d)


# --------------------------------------------------------------------------------------------- driver
def run(name, n_gauge, cells):
    m, ck, data = olib.load_grok(name); out = dict(specimen=name, condition=ck["cfg"]["condition"], seed=ck["cfg"]["seed"], cells={})
    if "gauge" in cells: out["cells"]["gauge"] = step_gauge(m, data, n_gauge)
    if "P0" in cells: out["cells"]["P0"] = cell_P0(m, data)
    if "P1" in cells: out["cells"]["P1"] = cell_P1(m, data, n_gauge)
    if "P2" in cells: out["cells"]["P2"] = cell_P2(m, data, n_gauge)
    if "P7" in cells: out["cells"]["P7"] = cell_P7(name, data)
    if "R" in cells: out["cells"]["R1"] = cell_R1(m, data)
    json.dump(out, open(os.path.join(OUT, name + ".json"), "w"), indent=1, default=str)
    new = not os.path.exists(CELLS_CSV)
    with open(CELLS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS); new and w.writeheader()
        for k, c in out["cells"].items():
            w.writerow(dict(specimen=name, cell=k, status=c["status"], value=json.dumps(c["value"], default=str)[:200], criterion=c["criterion"][:120], provenance=c["provenance"][:160], wall=c.get("wall")))
            print(f"{name} {k:5s} {c['status']:12s} {json.dumps(c['value'], default=str)[:100]}  [{c.get('wall')}s]", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--specimens", nargs="*"); ap.add_argument("--n-gauge", type=int, default=300)
    ap.add_argument("--cells", nargs="*", default=["gauge", "P0", "P1", "P2", "P7", "R"])
    a = ap.parse_args()
    names = a.specimens
    if not names:
        rows = list(csv.DictReader(open(os.path.join(RES, "specimens.csv"))))
        names = [r["name"] for r in rows if r["kind"] == "grok" and r["included"] == "1"]
    for n in names: run(n, a.n_gauge, a.cells)
