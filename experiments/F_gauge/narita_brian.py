"""F11 — real-model audit on GPT-2 small (SPEC §F11). One forward per renaming, all candidates at
once (``audit_quantities``). Writes results/F11_audit_table.csv, results/F11_identity.json."""
import os, sys, json, time, argparse
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import audit_quantities
from whiteboxtopo.core.persistence import compute_diagrams
from whiteboxtopo.topology import standardized_cloud
from special_week import ph_summary

R = os.path.join(os.path.dirname(__file__), "results")
LAYER, N_TOK, SEQ = 6, 2000, 125



def _owt_parquet():
    """OpenWebText-10k parquet: the local HF cache snapshot, else the copy shipped for the cluster in hpcc/data/ (WB_OWT_PARQUET overrides)."""
    import os
    cands = [os.environ.get("WB_OWT_PARQUET", ""),
             os.path.expanduser("~/.cache/huggingface/hub/datasets--stas--openwebtext-10k/snapshots/d0e7bd98eff9a555ba5ad44255cd64ea5df8c50b/plain_text/train/0000.parquet"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hpcc", "data", "openwebtext-10k.parquet")]
    for c in cands:
        if c and os.path.exists(c): return c
    raise FileNotFoundError("openwebtext-10k parquet not found (HF cache or hpcc/data/openwebtext-10k.parquet)")

def load_tokens(tok):
    p = os.path.join(R, "F11_tokens.npy")
    if os.path.exists(p): return np.load(p)
    df = pd.read_parquet(_owt_parquet())
    seqs = []
    for t in df.text:
        ids = tok(t)["input_ids"]
        for s in range(0, len(ids) - SEQ + 1, SEQ):
            seqs.append(ids[s:s + SEQ])
            if len(seqs) * SEQ >= N_TOK: break
        if len(seqs) * SEQ >= N_TOK: break
    toks = np.array(seqs, dtype=np.int64); np.save(p, toks); return toks


PH_SUB = np.random.default_rng(0).choice(N_TOK, 800, replace=False)   # AMENDMENTS #6


def ph20(cloud, seed=0):
    A = np.asarray(cloud, dtype=np.float64)[PH_SUB]; A = A - A.mean(0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    dg = compute_diagrams(A @ Vt[:20].T, maxdim=1, backend="ripser", thresh_percentile=95.0)
    return _summ(dg)


def _summ(dg):
    out = []
    for k in (0, 1):
        D = np.asarray(dg.dgms[k]).reshape(-1, 2); fin = np.isfinite(D[:, 1]); life = D[fin, 1] - D[fin, 0]
        p = life / life.sum() if life.sum() > 0 else np.zeros(0)
        out += [len(D), float(np.sum(life ** 2)), float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0]
    return np.array(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = "cuda"
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    toks = load_tokens(tok); T = torch.tensor(toks, device=dev)
    if a.smoke: T = T[:4]
    n_tok = T.numel()
    global PH_SUB
    PH_SUB = PH_SUB[PH_SUB < n_tok]; print("tokens", T.shape, flush=True)
    d = model.config.n_embd; H = model.config.n_head; dh = d // H
    blk = lambda m: m.transformer.h[LAYER]
    store = {}

    def quantities(m):
        cap = {}
        def hook_mlp(mod, inp, out): cap["mlp"] = out.detach()
        def hook_attn_in(mod, inp): cap["ln1"] = inp[0].detach()
        h1 = blk(m).mlp.act.register_forward_hook(hook_mlp)
        h2 = blk(m).attn.register_forward_pre_hook(hook_attn_in)
        with torch.no_grad():
            out = m(T, output_hidden_states=True)
        h1.remove(); h2.remove()
        logits = out.logits; resid = out.hidden_states[LAYER + 1][:, :, :].reshape(-1, d).float()
        mlp = cap["mlp"].reshape(-1, cap["mlp"].shape[-1]).float()                       # (n_tok, 3072) post-GELU
        # per-head outputs z (before W_O): recompute from the attention input
        with torch.no_grad():
            x = cap["ln1"]; qkv = blk(m).attn.c_attn(x); q, k, v = qkv.split(d, dim=2)
        B, Tn, _ = q.shape
        q = q.view(B, Tn, H, dh).transpose(1, 2); k = k.view(B, Tn, H, dh).transpose(1, 2); v = v.view(B, Tn, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        att = att.masked_fill(~torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=dev)), float("-inf")).softmax(-1)
        z = (att @ v).permute(0, 2, 1, 3).reshape(-1, H, dh).float().detach()            # (n_tok, H, dh)
        att = att.detach()
        # gradient x input for MLP hidden units under the next-token loss: parameters frozen, the MLP
        # activation is re-injected as a leaf so only the activation gradient is materialised (memory)
        for p_ in m.parameters(): p_.requires_grad_(False)
        gxi_acc = None
        for bi in range(0, T.shape[0], 4):
            cap2 = {}
            def hook_mlp_g(mod, inp, out):
                leaf = out.detach().requires_grad_(True); cap2["mlp"] = leaf; return leaf
            hg = blk(m).mlp.act.register_forward_hook(hook_mlp_g)
            Tb = T[bi:bi + 4]
            lg = m(Tb).logits; loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), Tb[:, 1:].reshape(-1), reduction="sum")
            (gr,) = torch.autograd.grad(loss, [cap2["mlp"]]); hg.remove()
            part = (cap2["mlp"] * gr).reshape(-1, cap2["mlp"].shape[-1]).sum(0).detach().float()
            gxi_acc = part if gxi_acc is None else gxi_acc + part
            del lg, loss, gr, cap2
        gxi = (gxi_acc / (T.shape[0] * (T.shape[1] - 1))).cpu().numpy()
        torch.cuda.empty_cache()
        Wfc = blk(m).mlp.c_fc.weight.detach().float(); Wpr = blk(m).mlp.c_proj.weight.detach().float()
        g1, b1 = blk(m).ln_1.weight.detach().float(), blk(m).ln_1.bias.detach().float()
        Wattn = blk(m).attn.c_attn.weight.detach().float(); Wo = blk(m).attn.c_proj.weight.detach().float()
        Wq_, Wk_, Wv_ = (g1[:, None] * Wattn).split(d, dim=1)     # LN-folded
        heads_qk, heads_ov = [], []
        for hh in range(H):
            sl = slice(hh * dh, (hh + 1) * dh)
            QK = Wq_[:, sl] @ Wk_[:, sl].T; OV = Wv_[:, sl] @ Wo[sl, :]
            heads_qk.append(torch.linalg.svdvals(QK)[:5].cpu().numpy()); heads_ov.append(torch.linalg.svdvals(OV)[:5].cpu().numpy())
        heads_qk = np.array(heads_qk); heads_ov = np.array(heads_ov)
        order = np.lexsort(heads_qk.T[::-1])
        mlp_np = mlp.cpu().numpy(); z_np = z.cpu().numpy(); resid_np = resid.cpu().numpy()
        Q = {
            "logits_sample": logits[:, :, :].reshape(-1, logits.shape[-1])[::97, ::1000].float().cpu().numpy().ravel(),
            "ph_resid_pca20": ph20(resid_np),
            "ph_mlp_pca20": ph20(mlp_np),
            "ph_mlp_std_pca20": ph20(standardized_cloud(mlp_np)),
            "ph_head_z_concat_pca20": ph20(z_np.reshape(len(z_np), -1)),
            "ph_head0_z": _summ(compute_diagrams(z_np[PH_SUB, 0, :], maxdim=1, backend="ripser", thresh_percentile=95.0)),
            "singular_values_Wfc": torch.linalg.svdvals(Wfc).cpu().numpy(),
            "singular_values_Wproj": torch.linalg.svdvals(Wpr).cpu().numpy(),
            "singular_values_Wfc_lnfolded": torch.linalg.svdvals(blk(m).ln_2.weight.detach().float()[:, None] * Wfc).cpu().numpy(),
            "per_unit_activation": mlp_np.mean(0),
            "per_unit_gradxinput": gxi,
            "topk10_units": np.sort(np.argsort(-np.abs(gxi))[:10]).astype(float),        # sorted unit indices
            "weight_norms_units": np.sqrt((Wfc ** 2).sum(0).cpu().numpy() + blk(m).mlp.c_fc.bias.detach().float().cpu().numpy() ** 2),
            "top_direction_mlp": (lambda v: v * np.sign(v[np.argmax(np.abs(v))]))(np.linalg.svd(mlp_np - mlp_np.mean(0), full_matrices=False)[2][0]),
            "per_head_norm": np.linalg.norm(z_np, axis=2).mean(0),
            "per_head_norm_sorted": np.sort(np.linalg.norm(z_np, axis=2).mean(0)),
            "head_QK_top5sv_sorted": heads_qk[order].ravel(),
            "head_OV_top5sv_sorted": heads_ov[order].ravel(),
            "attn_entropy_per_head": (-(att * torch.log(att + 1e-12)).sum(-1).mean((0, 2))).cpu().numpy(),
        }
        return Q

    from whiteboxtopo.symmetry.group import inverse_on_units
    key = ("mlp", LAYER)
    aligners = {q: (lambda g, v, k=key: inverse_on_units(g, k, v, axis=0)) for q in ("per_unit_activation", "per_unit_gradxinput", "weight_norms_units", "top_direction_mlp")}
    def topk_align(g, v):
        p = g.perms.get(key)
        return v if p is None else np.sort(np.asarray(p)[v.astype(int)]).astype(float)   # transformed unit j = base unit p[j]
    aligners["topk10_units"] = topk_align
    def head_align(g, v):
        p = g.perms.get(("head", LAYER))
        if p is None: return v
        inv = np.empty_like(p); inv[p] = np.arange(len(p)); return v[inv]
    aligners["per_head_norm"] = head_align; aligners["attn_entropy_per_head"] = head_align
    # identity check (fp32 floor) for each kind
    ident = {}
    with torch.no_grad():
        base_logits = model(T).logits.float()
        for kind in ("perm", "scale", "rot", "combined"):
            rng = np.random.default_rng(7); worst = 0.0
            for r in range(5 if not a.smoke else 2):
                g = S.random_element(model, rng, kind=kind, orthogonal=(r % 2 == 0))
                worst = max(worst, float((S.apply(model, g)(T).logits.float() - base_logits).abs().max()))
            ident[kind] = worst
    print("identity max|dlogit| fp32:", ident, flush=True)
    json.dump(dict(identity_max_abs_dlogit=ident, n_tokens=int(n_tok), layer=LAYER), open(os.path.join(R, "F11_identity.json"), "w"), indent=1)
    t0 = time.time()
    res = audit_quantities(quantities, model, n=a.n if not a.smoke else 3, kinds=("perm", "scale", "rot", "combined"), aligners=aligners,
                           orthogonal=False, verbose=True)
    rows = []
    for q, r in res.items():
        row = r.row(); row["layer"] = LAYER; rows.append(row)
        print(f"{q:32s} {r.verdict:9s} aligned={r.verdict_aligned}  " + " ".join(f"{k}={v:.2e}" for k, v in r.median_cv.items()), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(R, "F11_audit_table.csv" if not a.smoke else "F11_audit_table_smoke.csv"), index=False)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
