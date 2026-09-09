"""L7 — the auditor end-to-end on Gemma-2-9B (or Llama-3-8B) on HPCC, timing logged; what does not scale is written
into the timing file (handbook §11 L7). Mirrors benchmarks/l6_end_to_end_gpt2.py stage for stage with the
``symmetry.llama`` backend: load -> identity (bf16 floor per renaming kind) -> compact audit at one layer (residual / MLP
persistence summaries, RMSNorm-folded singular values, per-unit activation raw + aligned, per-head norms, attention
entropy) -> darkmatter on the K8 cache if present -> not-applicable / not-covered notes.
Usage:  python l7_end_to_end.py --model google/gemma-2-9b --layer 20 --n 6 --tokens 1000 --out results/l7 [--k8-cache results/k8/cache]
Needs: HF_TOKEN, >= 40 GB GPU (bf16 model + one renamed copy), whiteboxtopo."""
import os, sys, json, time, argparse, platform
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "K_darkmatter"))
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import audit_quantities
from whiteboxtopo.symmetry.group import inverse_on_units
from whiteboxtopo.core.persistence import compute_diagrams
from whiteboxtopo.darkmatter import residual_structure
sys.path.insert(0, os.path.join(ROOT, "benchmarks")); from l6_end_to_end_gpt2 import summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/gemma-2-9b"); ap.add_argument("--layer", type=int, default=20); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=1000); ap.add_argument("--out", default="results/l7"); ap.add_argument("--k8-cache", default=None); ap.add_argument("--dtype", default="bfloat16"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"; timing = {"host": platform.node(), "model": a.model, "dtype": a.dtype, "n_renamings_per_kind": a.n, "stages": {}, "does_not_scale": []}
    def stage(name, t0, **extra): timing["stages"][name] = dict(wall_s=round(time.time() - t0, 1), **extra); print(f"[L7] {name} {time.time()-t0:.1f}s", flush=True)
    T0 = time.time(); t = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import run_K1
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
    run_K1.CACHE = a.out; toks = run_K1.load_tokens(tok); seq = toks.shape[1]; toks = toks[: max(1, a.tokens // seq)]; T = torch.tensor(toks, device=dev); n_tok = int(T.numel())
    stage("load", t, n_tokens=n_tok, params=sum(p.numel() for p in model.parameters()))
    c = S.llama._cfg(model); H, dh, d, L = c["H"], c["dh"], c["d"], a.layer; blk = lambda m: m.model.layers[L]
    t = time.time(); ident = {}
    with torch.no_grad():
        base = model(T).logits.float()
        for kind in ("perm", "scale", "rot", "combined"):
            rng = np.random.default_rng(7); worst = 0.0
            for r in range(2):
                g = S.random_element(model, rng, kind=kind, orthogonal=(r % 2 == 0)); mg = S.apply(model, g)
                worst = max(worst, float((mg(T).logits.float() - base).abs().max())); del mg; torch.cuda.empty_cache()
            ident[kind] = worst
    stage("identity", t, max_abs_dlogit=ident, note=f"{a.dtype} floor; the SPEC's 1e-3 is an fp32 number (F11 measured 3-5e-3 on GPT-2 fp32)")
    if max(ident.values()) > 0.5: timing["does_not_scale"].append(f"identity deviation {max(ident.values()):.2f} in {a.dtype}: renamings are exact only in higher precision; rerun with --dtype float32 if memory allows")
    t = time.time(); sub = np.random.default_rng(0).choice(n_tok, min(500, n_tok), replace=False)
    def ph20(A):
        A = np.asarray(A, dtype=np.float64)[sub]; A = A - A.mean(0); Vt = np.linalg.svd(A, full_matrices=False)[2]; return summ(compute_diagrams(A @ Vt[:20].T, maxdim=1, backend="ripser", thresh_percentile=95.0))
    def quantities(m):
        cap = {}
        h1 = blk(m).mlp.down_proj.register_forward_pre_hook(lambda mod, i: cap.__setitem__("mlp", i[0].detach()))
        h2 = blk(m).self_attn.o_proj.register_forward_pre_hook(lambda mod, i: cap.__setitem__("z", i[0].detach()))
        with torch.no_grad(): out = m(T, output_hidden_states=True, output_attentions=True)
        h1.remove(); h2.remove()
        resid = out.hidden_states[L + 1].reshape(-1, d).float().cpu().numpy(); mlp = cap["mlp"].reshape(-1, cap["mlp"].shape[-1]).float().cpu().numpy(); z = cap["z"].reshape(-1, H, dh).float().cpu().numpy()
        att = out.attentions[L].float(); ent = (-(att * torch.log(att + 1e-12)).sum(-1).mean((0, 2))).cpu().numpy()
        Wd = blk(m).mlp.down_proj.weight.float(); w2 = blk(m).post_attention_layernorm.weight.float(); w2 = (1 + w2) if c["gemma"] else w2
        Wg = blk(m).mlp.gate_proj.weight.float()
        return {"logits_sample": out.logits.reshape(-1, out.logits.shape[-1])[::37, ::2000].float().cpu().numpy().ravel(), "ph_resid_pca20": ph20(resid), "ph_mlp_pca20": ph20(mlp),
                "singular_values_Wgate": torch.linalg.svdvals(Wg).cpu().numpy(), "singular_values_Wgate_rmsfolded": torch.linalg.svdvals(Wg * w2[None, :]).cpu().numpy(),
                "per_unit_activation": mlp.mean(0), "per_head_norm": np.linalg.norm(z, axis=2).mean(0), "per_head_norm_sorted": np.sort(np.linalg.norm(z, axis=2).mean(0)), "attn_entropy_per_head": ent}
    def head_align(g, v):
        p = g.perms.get(("head", L))
        if p is None: return v
        inv = np.empty_like(p); inv[p] = np.arange(len(p)); return v[inv]
    aligners = {"per_unit_activation": lambda g, v: inverse_on_units(g, ("mlp", L), v, axis=0), "per_head_norm": head_align, "attn_entropy_per_head": head_align}
    res = audit_quantities(quantities, model, n=a.n, kinds=("perm", "scale", "rot", "combined"), aligners=aligners, orthogonal=False, verbose=False)
    rows = [dict(r.row(), quantity=q, layer=L) for q, r in res.items()]; pd.DataFrame(rows).to_csv(os.path.join(a.out, "L7_audit.csv"), index=False)
    stage("audit", t, n_quantities=len(rows), verdicts={q: r.verdict for q, r in res.items()})
    if timing["stages"]["audit"]["wall_s"] > 3600: timing["does_not_scale"].append("audit > 1 h: one forward per renaming over the full model; restrict --tokens or audit one layer's block in isolation")
    del model; torch.cuda.empty_cache(); t = time.time()
    cp = os.path.join(a.k8_cache or "", f"L{L}.npz") if a.k8_cache else None
    if cp and os.path.exists(cp):
        E = np.load(cp)["E"]; idx = np.random.default_rng(0).choice(len(E), min(20000, len(E)), replace=False); rep = residual_structure(E[idx].astype(np.float64), topology=True, n_points=1500, n_boot=10)
        stage("darkmatter", t, n_outliers=int(rep.n_outliers), structured_fraction=float(rep.structured_fraction), verdict=rep.verdict, source=cp)
    else: stage("darkmatter", t, skipped="no K8 cache for this layer (run k8_frontier_residue.py first)")
    timing["stages"]["hidden"] = dict(wall_s=0.0, not_applicable="symmetry.hidden is ReLU-MLP-only")
    timing["not_covered"] = {"gluing": "EAP score matrix over a prompt set (E3) not run at 9B", "legibility": "node-instrumented small models only (E4)"}
    timing["total_wall_s"] = round(time.time() - T0, 1); timing["date"] = time.strftime("%Y-%m-%d %H:%M"); json.dump(timing, open(os.path.join(a.out, "L7_timing.json"), "w"), indent=1); print(timing)


if __name__ == "__main__":
    main()
