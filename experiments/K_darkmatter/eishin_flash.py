"""K1 — residuals e = x - SAE(x) for GPT-2 small, SAELens gpt2-small-res-jb, layers 3/6/9, 50k tokens
(SPEC §K1). Also saves Δlogits (clean − SAE-patched) on the top-2000 next-token vocabulary for K4/K5.
Run in ~/Whitebox/v2-programme/.venv-tl. Writes cache/L{L}.npz, results/K1_reconstruction.csv."""
import os, sys, json, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
LAYERS = [3, 6, 9]; N_SEQ = 400; SEQ_LEN = 126; BATCH = 8; TOPV = 2000
os.makedirs(CACHE, exist_ok=True); os.makedirs(RES, exist_ok=True)



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
    p = os.path.join(CACHE, "tokens.npy")
    if os.path.exists(p): return np.load(p)
    df = pd.read_parquet(_owt_parquet())
    seqs = []
    for t in df.text:
        ids = tok(t)["input_ids"]
        for s in range(0, len(ids) - (SEQ_LEN - 1) + 1, SEQ_LEN - 1):
            seqs.append([tok.bos_token_id] + ids[s:s + SEQ_LEN - 1])
            if len(seqs) >= N_SEQ: break
        if len(seqs) >= N_SEQ: break
    toks = np.array(seqs[:N_SEQ], dtype=np.int64); np.save(p, toks); return toks


def main():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    dev = "cuda"; t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2-small", device=dev)
    toks = load_tokens(model.tokenizer); tt = torch.as_tensor(toks); print("tokens", toks.shape, flush=True)
    # top-2000 next tokens by corpus frequency
    nt_all = toks[:, 1:].ravel(); vals, counts = np.unique(nt_all, return_counts=True); topV = np.sort(vals[np.argsort(-counts)[:TOPV]])
    np.save(os.path.join(CACHE, "topV.npy"), topV)
    rows = []
    for L in LAYERS:
        out = os.path.join(CACHE, f"L{L}.npz")
        if os.path.exists(out): print("skip", L); continue
        hook = f"blocks.{L}.hook_resid_pre"
        sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=dev); sae = sae[0] if isinstance(sae, tuple) else sae; sae.eval()
        cfg = {k: str(v) for k, v in vars(sae.cfg).items()} if hasattr(sae, "cfg") else {}
        Xs, Es, Ds, nts, sids, poss = [], [], [], [], [], []
        with torch.no_grad():
            for b in range(0, N_SEQ, BATCH):
                ids = tt[b:b + BATCH].to(dev)
                logits, cache = model.run_with_cache(ids, names_filter=hook)
                x = cache[hook]; r = sae.decode(sae.encode(x)); e = x - r
                patched = model.run_with_hooks(ids, fwd_hooks=[(hook, lambda act, hook: sae.decode(sae.encode(act)))])
                delta = (logits - patched)[:, 1:, :][..., torch.as_tensor(topV, device=dev)].float()
                Xs.append(x[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy()); Es.append(e[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy())
                Ds.append(delta.reshape(-1, TOPV).half().cpu().numpy())
                B_ = ids.shape[0]
                nts.append(np.concatenate([toks[b:b + B_, 2:], np.full((B_, 1), -1)], 1).reshape(-1))
                sids.append(np.repeat(np.arange(b, b + B_), SEQ_LEN - 1)); poss.append(np.tile(np.arange(1, SEQ_LEN), B_))
                del cache
        X = np.concatenate(Xs); E = np.concatenate(Es); D = np.concatenate(Ds)
        fvu = float((E ** 2).sum() / ((X - X.mean(0)) ** 2).sum()); mse = float((E ** 2).mean())
        np.savez(out, X=X, E=E, delta=D, next_token=np.concatenate(nts), seq=np.concatenate(sids), pos=np.concatenate(poss), fvu=fvu)
        rows.append(dict(layer=L, hook=hook, n_tokens=len(E), fvu=fvu, variance_explained=1 - fvu, mse=mse,
                         d_sae=cfg.get("d_sae"), published_note="SAELens gpt2-small-res-jb card: variance explained not machine-readable here; see FINDINGS",
                         elapsed_s=round(time.time() - t0)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "K1_reconstruction.csv"), index=False)
        del sae; torch.cuda.empty_cache()
    json.dump(dict(sae_cfg_last=cfg, n_seq=N_SEQ, seq_len=SEQ_LEN, layers=LAYERS), open(os.path.join(RES, "K1_meta.json"), "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
