"""L6 — the auditor end-to-end on GPT-2 small in one session, timing logged (handbook §11 L6).

Stages (each timed, each resumable through --skip):
  1. load        GPT-2 small (HF) + a 1 000-token OpenWebText probe (reuses experiments/F_gauge/results/F11_tokens.npy)
  2. identity    fp32 output deviation under each renaming kind of ``symmetry.gpt2`` (perm / scale / rot / combined)
  3. audit       ``invariants.audit_quantities`` on a compact candidate list at layer 6 (n renamings per kind): residual /
                 MLP persistence summaries, LN-folded singular values, per-unit activation (raw + gauge-aligned), per-head
                 output norms (raw / sorted), attention entropy
  4. darkmatter  ``darkmatter.residual_structure`` on a cached SAE residual (experiments/K_darkmatter/cache/L6.npz, E)
  5. hidden      ``symmetry.hidden`` is ReLU-MLP-only: recorded as NOT APPLICABLE to GPT-2 (documented, not skipped silently)
Not covered by L6 (documented in PAUSE_5): ``gluing`` needs an EAP graph over a prompt set (E3, hours); ``legibility``
is node-instrumented small models only (E4). Writes benchmarks/results/L6_timing.json, L6_audit.csv, L6_darkmatter.json.
Usage: python benchmarks/oguri_cap.py [--device cuda|cpu] [--n 8] [--tokens 1000]"""
import os, sys, json, time, argparse, platform
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
R = os.path.join(HERE, "results"); os.makedirs(R, exist_ok=True)
LAYER = 6


def summ(dg):
    out = []
    for k in (0, 1):
        D = np.asarray(dg.dgms[k]).reshape(-1, 2); fin = np.isfinite(D[:, 1]); life = D[fin, 1] - D[fin, 0]
        p = life / life.sum() if life.sum() > 0 else np.zeros(0)
        out += [len(D), float(np.sum(life ** 2)), float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0]
    return np.array(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1000); ap.add_argument("--ph-sub", type=int, default=500); a = ap.parse_args()
    import torch
    from whiteboxtopo import symmetry as S
    from whiteboxtopo.invariants import audit_quantities
    from whiteboxtopo.symmetry.group import inverse_on_units
    from whiteboxtopo.core.persistence import compute_diagrams
    from whiteboxtopo.darkmatter import residual_structure
    timing = {"host": platform.node(), "device": a.device, "n_renamings_per_kind": a.n, "tokens": a.tokens, "stages": {}}
    def stage(name, t0, **extra): timing["stages"][name] = dict(wall_s=round(time.time() - t0, 1), **extra); print(f"[L6] {name} {time.time()-t0:.1f}s {extra}", flush=True)
    T0 = time.time()
    # 1. load ------------------------------------------------------------------------------------------------
    t = time.time()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = a.device
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    tp = os.path.join(ROOT, "experiments", "F_gauge", "results", "F11_tokens.npy")
    if os.path.exists(tp):
        toks = np.load(tp)
    else:
        sys.path.insert(0, os.path.join(ROOT, "experiments", "F_gauge")); from narita_brian import load_tokens; toks = load_tokens(tok)
    seq = toks.shape[1]; toks = toks[: max(1, a.tokens // seq)]
    T = torch.tensor(toks, device=dev); n_tok = int(T.numel())
    stage("load", t, n_tokens=n_tok, seq_len=int(seq), n_seqs=int(T.shape[0]))
    d = model.config.n_embd; H = model.config.n_head; dh = d // H
    blk = lambda m: m.transformer.h[LAYER]
    # 2. identity --------------------------------------------------------------------------------------------
    t = time.time(); ident = {}
    with torch.no_grad():
        base = model(T).logits.float()
        for kind in ("perm", "scale", "rot", "combined"):
            rng = np.random.default_rng(7); worst = 0.0
            for r in range(3):
                g = S.random_element(model, rng, kind=kind, orthogonal=(r % 2 == 0))
                worst = max(worst, float((S.apply(model, g)(T).logits.float() - base).abs().max()))
            ident[kind] = worst
    stage("identity", t, max_abs_dlogit_fp32=ident)
    # 3. audit -----------------------------------------------------------------------------------------------
    t = time.time()
    sub = np.random.default_rng(0).choice(n_tok, min(a.ph_sub, n_tok), replace=False)
    def ph20(A):
        A = np.asarray(A, dtype=np.float64)[sub]; A = A - A.mean(0); Vt = np.linalg.svd(A, full_matrices=False)[2]
        return summ(compute_diagrams(A @ Vt[:20].T, maxdim=1, backend="ripser", thresh_percentile=95.0))
    def quantities(m):
        cap = {}
        h1 = blk(m).mlp.act.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", o.detach()))
        h2 = blk(m).attn.register_forward_pre_hook(lambda mod, i: cap.__setitem__("ln1", i[0].detach()))
        with torch.no_grad(): out = m(T, output_hidden_states=True)
        h1.remove(); h2.remove()
        resid = out.hidden_states[LAYER + 1].reshape(-1, d).float().cpu().numpy(); mlp = cap["mlp"].reshape(-1, cap["mlp"].shape[-1]).float().cpu().numpy()
        with torch.no_grad():
            x = cap["ln1"]; q, k, v = blk(m).attn.c_attn(x).split(d, dim=2)
            B, Tn, _ = q.shape
            q = q.view(B, Tn, H, dh).transpose(1, 2); k = k.view(B, Tn, H, dh).transpose(1, 2); v = v.view(B, Tn, H, dh).transpose(1, 2)
            att = ((q @ k.transpose(-1, -2)) / np.sqrt(dh)).masked_fill(~torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=dev)), float("-inf")).softmax(-1)
            z = (att @ v).permute(0, 2, 1, 3).reshape(-1, H, dh).float().cpu().numpy()
            ent = (-(att * torch.log(att + 1e-12)).sum(-1).mean((0, 2))).cpu().numpy()
        Wfc = blk(m).mlp.c_fc.weight.detach().float()
        return {
            "logits_sample": out.logits.reshape(-1, out.logits.shape[-1])[::37, ::500].float().cpu().numpy().ravel(),
            "ph_resid_pca20": ph20(resid), "ph_mlp_pca20": ph20(mlp),
            "singular_values_Wfc": torch.linalg.svdvals(Wfc).cpu().numpy(),
            "singular_values_Wfc_lnfolded": torch.linalg.svdvals(blk(m).ln_2.weight.detach().float()[:, None] * Wfc).cpu().numpy(),
            "per_unit_activation": mlp.mean(0),
            "per_head_norm": np.linalg.norm(z, axis=2).mean(0), "per_head_norm_sorted": np.sort(np.linalg.norm(z, axis=2).mean(0)),
            "attn_entropy_per_head": ent,
        }
    key = ("mlp", LAYER)
    def head_align(g, v):
        p = g.perms.get(("head", LAYER))
        if p is None: return v
        inv = np.empty_like(p); inv[p] = np.arange(len(p)); return v[inv]
    aligners = {"per_unit_activation": lambda g, v: inverse_on_units(g, key, v, axis=0), "per_head_norm": head_align, "attn_entropy_per_head": head_align}
    res = audit_quantities(quantities, model, n=a.n, kinds=("perm", "scale", "rot", "combined"), aligners=aligners, orthogonal=False, verbose=False)
    rows = []
    for qn, r in res.items():
        row = r.row(); row["quantity"] = qn; row["layer"] = LAYER; rows.append(row)
        print(f"  {qn:30s} {r.verdict:9s} aligned={r.verdict_aligned}  " + " ".join(f"{k}={v:.1e}" for k, v in r.median_cv.items()), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, "L6_audit.csv"), index=False)
    stage("audit", t, n_quantities=len(rows), verdicts={qn: r.verdict for qn, r in res.items()})
    del model; 
    if dev == "cuda": torch.cuda.empty_cache()
    # 4. darkmatter ------------------------------------------------------------------------------------------
    t = time.time(); dm_path = os.path.join(ROOT, "experiments", "K_darkmatter", "cache", f"L{LAYER}.npz")
    if os.path.exists(dm_path):
        Z = np.load(dm_path); E = Z["E"]; idx = np.random.default_rng(0).choice(len(E), min(20000, len(E)), replace=False)
        rep = residual_structure(E[idx].astype(np.float64), topology=True, n_points=1500, n_boot=10)
        dm = dict(n_outliers=int(rep.n_outliers), structured_fraction=float(rep.structured_fraction), verdict=rep.verdict,
                  spectral_test_valid=bool(rep.spectral_test_valid), tc_ratio=float(getattr(rep, "tc_ratio", float("nan"))), n_tokens=int(len(idx)), source=os.path.relpath(dm_path, ROOT))
        json.dump(dm, open(os.path.join(R, "L6_darkmatter.json"), "w"), indent=1)
        stage("darkmatter", t, **dm)
    else:
        stage("darkmatter", t, skipped="no cached SAE residual (run experiments/K_darkmatter/eishin_flash.py in the TL venv)")
    # 5. hidden (not applicable) ------------------------------------------------------------------------------
    timing["stages"]["hidden"] = dict(wall_s=0.0, not_applicable="symmetry.hidden is ReLU-MLP-only (F3); no transformer fiber-dimension backend ships")
    timing["not_covered"] = {"gluing": "needs an EAP score matrix over a prompt set (E3: hours on GPT-2 small); see benchmarks/GLUING.md",
                             "legibility": "node-instrumented small models only (E4, N <= 300); see benchmarks/PHASE6.md"}
    timing["total_wall_s"] = round(time.time() - T0, 1); timing["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(timing, open(os.path.join(R, "L6_timing.json"), "w"), indent=1)
    print(f"[L6] total {timing['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
