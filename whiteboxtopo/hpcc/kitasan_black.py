"""F12 — frontier audit (HPCC): Llama-3-8B or Gemma-2-9B, 3 layers, F11's candidate list, checkpointed per
(layer, kind, sample) with --resume. Usage:
  python kitasan_black.py --model meta-llama/Meta-Llama-3-8B --layers 8 16 24 --n 20 --out results/f12_llama3_8b --resume
Needs: HF token in the environment (HF_TOKEN), a GPU with >= 40 GB for bf16 8B/9B with a renamed copy (or --cpu-copy to
keep the renamed copy on CPU and stream layers), whiteboxtopo installed (pip install -e ~/Whitebox/whitebox-v2).
Renamings: symmetry.llama backend (mlp perms, KV/Q head perms, V/O rotations, RoPE-commuting Q/K per-plane phases + scales,
RMSNorm rescales). Candidates per layer: persistence summaries (PCA-20, 800-token subsample) of the residual stream, the MLP
hidden activations (post-gate) and per-head outputs; singular values of gate/up/down (raw and RMSNorm-folded); per-unit
activation, per-unit gradient x input for the next-token loss, top-10 units, unit weight norms, top principal direction;
per-head output norms (raw / sorted); rotation-invariant per-head forms (top-5 singular values of the folded QK and OV maps, sorted)."""
import os, sys, json, time, argparse
import numpy as np, torch, pandas as pd
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants.check import per_component_cv, INVARIANT_CV
from whiteboxtopo.core.persistence import compute_diagrams
from whiteboxtopo.topology import standardized_cloud
from whiteboxtopo.symmetry.group import inverse_on_units


def summ(dg):
    out = []
    for k in (0, 1):
        D = np.asarray(dg.dgms[k]).reshape(-1, 2); fin = np.isfinite(D[:, 1]); life = D[fin, 1] - D[fin, 0]
        p = life / life.sum() if life.sum() > 0 else np.zeros(0)
        out += [len(D), float(np.sum(life ** 2)), float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0]
    return np.array(out)


def ph20(A, sub):
    A = np.asarray(A, dtype=np.float64)[sub]; A = A - A.mean(0); Vt = np.linalg.svd(A, full_matrices=False)[2]
    return summ(compute_diagrams(A @ Vt[:20].T, maxdim=1, backend="ripser", thresh_percentile=95.0))


def make_quantities(model, T, layer, sub):
    c = S.llama._cfg(model); H, KV, dh, d, G = c["H"], c["KV"], c["dh"], c["d"], c["group"]
    def q(m):
        blk = m.model.layers[layer]; cap = {}
        h1 = blk.mlp.down_proj.register_forward_pre_hook(lambda mod, i: cap.__setitem__("mlp", i[0].detach()))
        h2 = blk.self_attn.o_proj.register_forward_pre_hook(lambda mod, i: cap.__setitem__("heads", i[0].detach()))
        with torch.no_grad(): out = m(T, output_hidden_states=True)
        h1.remove(); h2.remove()
        resid = out.hidden_states[layer + 1].reshape(-1, d).float().cpu().numpy(); mlp = cap["mlp"].reshape(-1, c["m"]).float().cpu().numpy()
        z = cap["heads"].reshape(-1, H, dh).float().cpu().numpy()
        for p in m.parameters(): p.requires_grad_(False)
        gx = None
        for bi in range(0, T.shape[0], 2):
            cap2 = {}
            def hk(mod, i):
                leaf = i[0].detach().requires_grad_(True); cap2["a"] = leaf; return (leaf,)
            hg = blk.mlp.down_proj.register_forward_pre_hook(hk); Tb = T[bi:bi + 2]
            lg = m(Tb).logits; loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), Tb[:, 1:].reshape(-1), reduction="sum")
            (gr,) = torch.autograd.grad(loss, [cap2["a"]]); hg.remove()
            part = (cap2["a"] * gr).reshape(-1, c["m"]).sum(0).float(); gx = part if gx is None else gx + part; del lg, loss, gr
        gx = (gx / (T.shape[0] * (T.shape[1] - 1))).cpu().numpy()
        Wg, Wu, Wd = blk.mlp.gate_proj.weight.float(), blk.mlp.up_proj.weight.float(), blk.mlp.down_proj.weight.float()
        w2 = blk.post_attention_layernorm.weight.float(); w2 = (1 + w2) if c["gemma"] else w2
        w1 = blk.input_layernorm.weight.float(); w1 = (1 + w1) if c["gemma"] else w1
        Wq, Wk, Wv, Wo = blk.self_attn.q_proj.weight.float(), blk.self_attn.k_proj.weight.float(), blk.self_attn.v_proj.weight.float(), blk.self_attn.o_proj.weight.float()
        qk, ov = [], []
        for h in range(H):
            kv = h // G; Qh = Wq[h * dh:(h + 1) * dh] * w1[None]; Kh = Wk[kv * dh:(kv + 1) * dh] * w1[None]; Vh = Wv[kv * dh:(kv + 1) * dh] * w1[None]; Oh = Wo[:, h * dh:(h + 1) * dh]
            qk.append(torch.linalg.svdvals(Qh.T @ Kh)[:5].cpu().numpy()); ov.append(torch.linalg.svdvals(Oh @ Vh)[:5].cpu().numpy())
        qk = np.array(qk); ov = np.array(ov); order = np.lexsort(qk.T[::-1])
        return {"ph_resid_pca20": ph20(resid, sub), "ph_mlp_pca20": ph20(mlp, sub), "ph_mlp_std_pca20": ph20(standardized_cloud(mlp), sub),
                "ph_heads_concat_pca20": ph20(z.reshape(len(z), -1), sub),
                "sv_gate": torch.linalg.svdvals(Wg).cpu().numpy(), "sv_up": torch.linalg.svdvals(Wu).cpu().numpy(), "sv_down": torch.linalg.svdvals(Wd).cpu().numpy(),
                "sv_gate_normfolded": torch.linalg.svdvals(Wg * w2[None]).cpu().numpy(),
                "per_unit_activation": mlp.mean(0), "per_unit_gradxinput": gx, "topk10_units": np.sort(np.argsort(-np.abs(gx))[:10]).astype(float),
                "weight_norms_units": np.sqrt((Wg ** 2).sum(1).cpu().numpy()),
                "top_direction_mlp": (lambda v: v * np.sign(v[np.argmax(np.abs(v))]))(np.linalg.svd((mlp - mlp.mean(0))[sub], full_matrices=False)[2][0]),
                "per_head_norm": np.linalg.norm(z, axis=2).mean(0), "per_head_norm_sorted": np.sort(np.linalg.norm(z, axis=2).mean(0)),
                "head_QK_top5sv_sorted": qk[order].ravel(), "head_OV_top5sv_sorted": ov[order].ravel()}
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--layers", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", required=True); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-tokens", type=int, default=2000); ap.add_argument("--dtype", default="bfloat16"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); ck = os.path.join(a.out, "checkpoints"); os.makedirs(ck, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).cuda().eval()
    from datasets import load_dataset
    ds = load_dataset("stas/openwebtext-10k", split="train"); seqs = []
    for t in ds["text"]:
        ids = tok(t)["input_ids"]
        for s in range(0, len(ids) - 124, 125): seqs.append(ids[s:s + 125])
        if len(seqs) * 125 >= a.n_tokens: break
    T = torch.tensor(seqs, device="cuda"); n_tok = T.numel(); sub = np.random.default_rng(0).choice(n_tok, 800, replace=False)
    ident = {}
    with torch.no_grad():
        base = model(T).logits.float()
        for kind in ("perm", "scale", "rot", "combined"):
            g = S._backend(model).sample(model, np.random.default_rng(7), kind=kind, orthogonal=False, layers=a.layers)
            mg = S.apply(model, g); ident[kind] = float((mg(T).logits.float() - base).abs().max()); del mg; torch.cuda.empty_cache()
    json.dump(dict(identity=ident, dtype=a.dtype, n_tokens=int(n_tok)), open(os.path.join(a.out, "identity.json"), "w"), indent=1); print("identity", ident, flush=True)
    rows = []
    for layer in a.layers:
        qf = make_quantities(model, T, layer, sub); base_q = {k: np.asarray(v, dtype=np.float64) for k, v in qf(model).items()}
        key = ("mlp", layer); samples = {}
        for kind in ("perm", "scale", "rot", "combined"):
            rng = np.random.default_rng(1000 + hash(kind) % 100); samples[kind] = {q: [base_q[q]] for q in base_q}; al = {q: [base_q[q]] for q in base_q}
            for r in range(1, a.n):
                cp = os.path.join(ck, f"L{layer}_{kind}_{r}.npz")
                if a.resume and os.path.exists(cp):
                    z = np.load(cp, allow_pickle=True); vals = {q: z[q] for q in base_q}
                else:
                    g = S._backend(model).sample(model, rng, kind=kind, orthogonal=False, layers=[layer]); mg = S.apply(model, g)
                    vals = {q: np.asarray(v, dtype=np.float64) for q, v in qf(mg).items()}; del mg; torch.cuda.empty_cache()
                    for q in ("per_unit_activation", "per_unit_gradxinput", "weight_norms_units", "top_direction_mlp"):
                        vals[q + "__aligned"] = inverse_on_units(g, key, vals[q], axis=0)
                    p = g.perms.get(key); vals["topk10_units__aligned"] = np.sort(np.asarray(p)[vals["topk10_units"].astype(int)]).astype(float) if p is not None else vals["topk10_units"]
                    np.savez(cp, **vals)
                for q in base_q: samples[kind][q].append(vals[q])
                for q in base_q:
                    if q + "__aligned" in vals: al[q].append(vals[q + "__aligned"])
                print(f"L{layer} {kind} {r}/{a.n-1}", flush=True)
            for q in base_q:
                cv = per_component_cv(np.stack(samples[kind][q])); row = dict(layer=layer, metric=q, kind=kind, median_cv=float(np.median(cv)), frac_inv=float(np.mean(cv < INVARIANT_CV)))
                if len(al[q]) > 1: cva = per_component_cv(np.stack(al[q])); row["median_cv_aligned"] = float(np.median(cva))
                rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "F12_audit_table.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
