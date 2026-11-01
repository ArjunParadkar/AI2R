"""E6 extraction: resid_pre activations, SAE errors, and streaming Test-3 probes for GPT-2 small.
Writes E6/cache/L{L}.npz (X, E fp32, seq ids, pos, next_token) and E6/results/e6_probe.csv.
GPU budget ≤ 3 GB. See SPEC.md."""
import os, sys, time, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6lib.probe import RidgeAccumulator, R2Evaluator, ridge_onehot_probe
from e6lib.rmt import feature_shuffle

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
LAYERS = [2, 6, 10]; N_SEQ = 512; SEQ_LEN = 128; BATCH = 16; DBATCH = 8
os.makedirs(CACHE, exist_ok=True); os.makedirs(RES, exist_ok=True)


def load_tokens(tok):
    p = os.path.join(CACHE, "tokens.npy")
    if os.path.exists(p):
        return np.load(p), json.load(open(os.path.join(CACHE, "tokens_meta.json")))
    from datasets import load_dataset
    src = "stas/openwebtext-10k"
    try:
        ds = load_dataset(src, revision="refs/convert/parquet", split="train")
        texts = (d["text"] for d in ds)
    except Exception as e:
        print("openwebtext unavailable:", e); src = "wikitext-103-raw-v1/test"
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test"); texts = (d["text"] for d in ds)
    seqs, n_docs = [], 0
    for t in texts:
        n_docs += 1
        ids = tok(t)["input_ids"]
        for s in range(0, len(ids) - (SEQ_LEN - 1) + 1, SEQ_LEN - 1):
            seqs.append([tok.bos_token_id] + ids[s:s + SEQ_LEN - 1])
            if len(seqs) >= N_SEQ: break
        if len(seqs) >= N_SEQ: break
    toks = np.array(seqs[:N_SEQ], dtype=np.int64); meta = dict(source=src, n_docs_scanned=n_docs, n_seq=len(toks), seq_len=SEQ_LEN)
    np.save(p, toks); json.dump(meta, open(os.path.join(CACHE, "tokens_meta.json"), "w")); return toks, meta


def main():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    dev = "cuda"
    t0 = time.time()
    model = HookedTransformer.from_pretrained_no_processing("gpt2-small", device=dev, center_writing_weights=True)  # SPEC amendment 1
    toks, meta = load_tokens(model.tokenizer); print("tokens", toks.shape, meta)
    tt = torch.as_tensor(toks)
    hold = np.arange(N_SEQ) % 5 == 0    # held-out sequences (SPEC)
    rows = []
    for L in LAYERS:
        hook = f"blocks.{L}.hook_resid_pre"
        sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=dev)
        sae = sae[0] if isinstance(sae, tuple) else sae
        sae.eval()
        cfg = {k: str(v) for k, v in vars(sae.cfg).items()} if hasattr(sae, "cfg") else {}
        Xs, Es, next_tok, seq_id, pos = [], [], [], [], []
        # ---- pass 0: activations + errors
        with torch.no_grad():
            for b in range(0, N_SEQ, BATCH):
                _, cache = model.run_with_cache(tt[b:b + BATCH].to(dev), names_filter=hook)
                x = cache[hook]                      # (B, T, d)
                r = sae.decode(sae.encode(x)); e = x - r
                Xs.append(x[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy())
                Es.append(e[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy())
                B_ = x.shape[0]
                next_tok.append(np.concatenate([toks[b:b + B_, 2:], np.full((B_, 1), -1)], 1).reshape(-1))
                seq_id.append(np.repeat(np.arange(b, b + B_), SEQ_LEN - 1)); pos.append(np.tile(np.arange(1, SEQ_LEN), B_))
                del cache
        X = np.concatenate(Xs); E = np.concatenate(Es); nt = np.concatenate(next_tok); sid = np.concatenate(seq_id); ps = np.concatenate(pos)
        fvu = float((E ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
        print(f"L{L}: X {X.shape} FVU={fvu:.4f} ({time.time()-t0:.0f}s)")
        np.savez(os.path.join(CACHE, f"L{L}.npz"), X=X, E=E, next_token=nt, seq=sid, pos=ps, fvu=fvu)
        # ---- Test 3a: streaming ridge E→Δ, shuffled-E→Δ, X→Δ ; Δ = logits(clean) − logits(patched)
        d = X.shape[1]; V = model.cfg.d_vocab
        mu_e, sd_e = E.mean(0), E.std(0) + 1e-8; mu_x, sd_x = X.mean(0), X.std(0) + 1e-8
        # shuffled E: column permutation over the TRAIN tokens only; aligned to token index via a global permuted copy
        tr_idx = np.flatnonzero(~hold[sid]); Esh = E.copy(); Esh[tr_idx] = feature_shuffle(E[tr_idx], seed=L)
        accs = {k: RidgeAccumulator(d, V, lam=1.0, device=dev, dtype=torch.float32) for k in ("E", "Eshuf", "X")}
        def patched_logits(batch_ids):
            def hk(act, hook):
                return sae.decode(sae.encode(act))
            return model.run_with_hooks(batch_ids, fwd_hooks=[(hook, hk)])
        def delta_batches(which):
            with torch.no_grad():
                for b in range(0, N_SEQ, DBATCH):
                    sel = np.arange(b, min(b + DBATCH, N_SEQ)); sel = sel[hold[sel] == which]
                    if len(sel) == 0: continue
                    ids = tt[sel].to(dev)
                    delta = (model(ids) - patched_logits(ids))[:, 1:].reshape(-1, V).float()   # drop BOS
                    ti = np.concatenate([np.arange(s * (SEQ_LEN - 1), (s + 1) * (SEQ_LEN - 1)) for s in sel])
                    yield ti, delta
        for ti, delta in delta_batches(False):
            accs["E"].add(torch.as_tensor((E[ti] - mu_e) / sd_e), delta)
            accs["Eshuf"].add(torch.as_tensor((Esh[ti] - mu_e) / sd_e), delta)
            accs["X"].add(torch.as_tensor((X[ti] - mu_x) / sd_x), delta)
        for a in accs.values(): a.fit()
        evs = {k: R2Evaluator(accs[k].train_mean()) for k in accs}
        dnorm = []
        for ti, delta in delta_batches(True):
            evs["E"].add(delta, accs["E"].predict(torch.as_tensor((E[ti] - mu_e) / sd_e)))
            evs["Eshuf"].add(delta, accs["Eshuf"].predict(torch.as_tensor((Esh[ti] - mu_e) / sd_e)))
            evs["X"].add(delta, accs["X"].predict(torch.as_tensor((X[ti] - mu_x) / sd_x)))
            dnorm.append(delta.norm(dim=1).cpu().numpy())
        del accs; torch.cuda.empty_cache()
        r2 = {k: evs[k].r2 for k in evs}; print(f"L{L} probe R2 (held-out):", r2)
        # ---- Test 3b: next-token identity, top-V=2000 frequent next tokens in train
        valid = nt >= 0; tr = valid & ~hold[sid]; te = valid & hold[sid]
        vals, counts = np.unique(nt[tr], return_counts=True); topV = vals[np.argsort(-counts)[:2000]]
        remap = -np.ones(V, dtype=np.int64); remap[topV] = np.arange(len(topV))
        trm = tr & (remap[np.maximum(nt, 0)] >= 0); tem = te & (remap[np.maximum(nt, 0)] >= 0)
        cov_tr, cov_te = float(trm.sum() / tr.sum()), float(tem.sum() / te.sum())
        ytr, yte = remap[nt[trm]], remap[nt[tem]]
        maj = float(np.mean(yte == np.bincount(ytr).argmax()))
        res_nt = {}
        for k, Z in (("E", (E - mu_e) / sd_e), ("Eshuf", (Esh - mu_e) / sd_e), ("X", (X - mu_x) / sd_x)):
            res_nt[k] = ridge_onehot_probe(Z[trm], ytr, Z[tem], yte, len(topV), lam=1.0)
        print(f"L{L} next-token:", res_nt, "majority", maj, "coverage", cov_tr, cov_te)
        rows.append(dict(layer=L, fvu=fvu, n_tokens=len(E), n_train=int((~hold[sid]).sum()), n_test=int(hold[sid].sum()),
                         r2_E=r2["E"], r2_Eshuf=r2["Eshuf"], r2_X=r2["X"], mean_delta_norm=float(np.concatenate(dnorm).mean()),
                         nt_top1_E=res_nt["E"]["top1"], nt_top5_E=res_nt["E"]["top5"], nt_top1_Eshuf=res_nt["Eshuf"]["top1"],
                         nt_top5_Eshuf=res_nt["Eshuf"]["top5"], nt_top1_X=res_nt["X"]["top1"], nt_top5_X=res_nt["X"]["top5"],
                         nt_majority=maj, nt_coverage_train=cov_tr, nt_coverage_test=cov_te, n_classes=len(topV),
                         sae_release="gpt2-small-res-jb", sae_hook=hook, d_sae=cfg.get("d_sae"), elapsed_s=time.time() - t0))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "e6_probe.csv"), index=False)
        del sae; torch.cuda.empty_cache()
    json.dump(dict(tokens=meta, sae_cfg_last=cfg), open(os.path.join(RES, "e6_extract_meta.json"), "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
