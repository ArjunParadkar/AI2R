"""K8 — frontier residue: Gemma-2-9B + Gemma Scope (residual, canonical) at three layers, then K2–K5 exactly as on
GPT-2 small (HPCC). Produces the K1 cache format (X, E, delta on the top-2000 next tokens, next_token, seq, pos, fvu) so
``experiments/K_darkmatter/smart_falcon.py``'s k2 / k3 / k4_k5 run unchanged on it.

Usage:
  python biwa_hayahide.py --model google/gemma-2-9b --release gemma-scope-9b-pt-res-canonical --layers 9 20 31 \
      --width 131k --n-seq 400 --out results/k8 --resume
Needs: HF_TOKEN, >= 40 GB GPU (bf16 9B + SAE), transformer_lens + sae_lens, whiteboxtopo. Checkpoint per layer
(cache npz + K1-style row); K2–K5 rows are recomputed from the cache at the end (cheap, CPU)."""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
KD = os.path.join(ROOT, "experiments", "K_darkmatter"); sys.path.insert(0, KD)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/gemma-2-9b"); ap.add_argument("--release", default="gemma-scope-9b-pt-res-canonical")
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 20, 31]); ap.add_argument("--width", default="131k"); ap.add_argument("--n-seq", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=126); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--topv", type=int, default=2000)
    ap.add_argument("--out", default="results/k8"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--dtype", default="bfloat16"); a = ap.parse_args()
    cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True); dev = "cuda"
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import eishin_flash, smart_falcon
    eishin_flash.N_SEQ, eishin_flash.SEQ_LEN, eishin_flash.CACHE = a.n_seq, a.seq_len, cache
    model = HookedTransformer.from_pretrained_no_processing(a.model, device=dev, dtype=getattr(torch, a.dtype))
    toks = eishin_flash.load_tokens(model.tokenizer); tt = torch.as_tensor(toks)
    nt_all = toks[:, 1:].ravel(); vals, counts = np.unique(nt_all, return_counts=True); topV = np.sort(vals[np.argsort(-counts)[:a.topv]]); np.save(os.path.join(cache, "topV.npy"), topV)
    rows_p = os.path.join(a.out, "K8_reconstruction.csv"); rows = pd.read_csv(rows_p).to_dict("records") if (a.resume and os.path.exists(rows_p)) else []
    for L in a.layers:
        out = os.path.join(cache, f"L{L}.npz")
        if os.path.exists(out): print("skip", L, flush=True); continue
        t0 = time.time(); hook = f"blocks.{L}.hook_resid_post"
        sae = SAE.from_pretrained(a.release, f"layer_{L}/width_{a.width}/canonical", device=dev); sae = sae[0] if isinstance(sae, tuple) else sae; sae.eval()
        Xs, Es, Ds, nts, sids, poss = [], [], [], [], [], []
        with torch.no_grad():
            for b in range(0, a.n_seq, a.batch):
                ids = tt[b:b + a.batch].to(dev); logits, c = model.run_with_cache(ids, names_filter=hook); x = c[hook]
                r = sae.decode(sae.encode(x.to(sae.dtype))).to(x.dtype); e = x - r
                patched = model.run_with_hooks(ids, fwd_hooks=[(hook, lambda act, hook: sae.decode(sae.encode(act.to(sae.dtype))).to(act.dtype))])
                delta = (logits - patched)[:, 1:, :][..., torch.as_tensor(topV, device=dev)].float()
                Xs.append(x[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy()); Es.append(e[:, 1:].reshape(-1, x.shape[-1]).float().cpu().numpy())
                Ds.append(delta.reshape(-1, a.topv).half().cpu().numpy()); B_ = ids.shape[0]
                nts.append(np.concatenate([toks[b:b + B_, 2:], np.full((B_, 1), -1)], 1).reshape(-1)); sids.append(np.repeat(np.arange(b, b + B_), a.seq_len - 1)); poss.append(np.tile(np.arange(1, a.seq_len), B_))
                del c, logits, patched
        X = np.concatenate(Xs); E = np.concatenate(Es); D = np.concatenate(Ds); fvu = float((E ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
        np.savez(out, X=X, E=E, delta=D, next_token=np.concatenate(nts), seq=np.concatenate(sids), pos=np.concatenate(poss), fvu=fvu)
        rows.append(dict(model=a.model, layer=L, hook=hook, width=a.width, n_tokens=len(E), fvu=fvu, variance_explained=1 - fvu, mse=float((E ** 2).mean()), wall_s=round(time.time() - t0)))
        pd.DataFrame(rows).to_csv(rows_p, index=False); print(rows[-1], flush=True); del sae; torch.cuda.empty_cache()
    del model; torch.cuda.empty_cache()
    # K2–K5 on the cache, using the laptop analysis code unchanged (its module globals redirected to this run's dirs)
    smart_falcon.CACHE, smart_falcon.R, smart_falcon.LAYERS = cache, a.out, a.layers
    os.makedirs(os.path.join(a.out, "fig"), exist_ok=True)
    k2, k3, k45 = [], [], []
    for L in a.layers:
        Z = np.load(os.path.join(cache, f"L{L}.npz")); E, D, seq = Z["E"].astype(np.float64), Z["delta"].astype(np.float32), Z["seq"]
        k2.append(smart_falcon.k2(L, E, seq)); k3.append(smart_falcon.k3(L, E)); r45 = smart_falcon.k4_k5(L, E, D, seq); k45.append(r45)
        print("K2", k2[-1], "\nK3", k3[-1], "\nK4/K5", r45, flush=True)
    for name, rs in (("K2_spectral", k2), ("K3_topology", k3), ("K4_K5", k45)):
        pd.DataFrame([r if isinstance(r, dict) else (r[0] if isinstance(r, (list, tuple)) and r and isinstance(r[0], dict) else {"raw": str(r)}) for r in rs]).to_csv(os.path.join(a.out, f"K8_{name}.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
