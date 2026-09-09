"""K7 — dark-matter width scaling on the Gemma Scope width ladder (HPCC). **REPLICATION, not a novelty claim**:
Engels, Smith & Tegmark (arXiv 2410.14670) already show the nonlinear SAE error does not shrink with width; K7 re-asks
the question with the spectral statistic (K2: MP/Tracy-Widom outlier count of the residual correlation matrix and the
structured fraction, via ``whiteboxtopo.darkmatter.residual_structure``) and reports agreement or disagreement with them.
NOVELTY_BOARD.md #4.

Usage:
  python k7_width_scaling.py --model google/gemma-2-2b --release gemma-scope-2b-pt-res-canonical --layer 12 \
      --widths 16k 32k 65k 131k 262k 524k 1m --n-seq 400 --out results/k7 --resume
Needs: HF_TOKEN (Gemma licence), transformer_lens + sae_lens (the laptop's ~/Whitebox/v2-programme/.venv-tl recipe),
whiteboxtopo. Checkpoint per width (results/k7/L{layer}_w{width}.npz + one CSV row); --resume skips finished widths.
Tokens: the same OpenWebText-10k slice as K1 (``experiments/K_darkmatter/run_K1.py::load_tokens``), regenerated on the
cluster if cache/tokens.npy is absent (the parquet path is the HF datasets cache)."""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "K_darkmatter"))
from whiteboxtopo.darkmatter import residual_structure


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/gemma-2-2b"); ap.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--layer", type=int, default=12); ap.add_argument("--widths", nargs="+", default=["16k", "32k", "65k", "131k", "262k", "524k", "1m"])
    ap.add_argument("--n-seq", type=int, default=400); ap.add_argument("--seq-len", type=int, default=126); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="results/k7"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--dtype", default="bfloat16"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import run_K1
    run_K1.N_SEQ, run_K1.SEQ_LEN = a.n_seq, a.seq_len
    model = HookedTransformer.from_pretrained_no_processing(a.model, device=dev, dtype=getattr(torch, a.dtype))
    toks = run_K1.load_tokens(model.tokenizer); tt = torch.as_tensor(toks); hook = f"blocks.{a.layer}.hook_resid_post"
    csv_p = os.path.join(a.out, "K7_width_scaling.csv"); rows = pd.read_csv(csv_p).to_dict("records") if (a.resume and os.path.exists(csv_p)) else []
    done = {r["width"] for r in rows}
    for w in a.widths:
        if w in done: print("skip", w, flush=True); continue
        t0 = time.time(); sae_id = f"layer_{a.layer}/width_{w}/canonical"
        sae = SAE.from_pretrained(a.release, sae_id, device=dev); sae = sae[0] if isinstance(sae, tuple) else sae; sae.eval()
        Es = []
        with torch.no_grad():
            for b in range(0, a.n_seq, a.batch):
                ids = tt[b:b + a.batch].to(dev); _, cache = model.run_with_cache(ids, names_filter=hook); x = cache[hook].float()
                e = x - sae.decode(sae.encode(x.to(sae.dtype))).float(); Es.append(e[:, 1:].reshape(-1, x.shape[-1]).cpu().numpy()); del cache
        E = np.concatenate(Es).astype(np.float64); fvu = float((E ** 2).sum() / ((E - E.mean(0)) ** 2).sum()) if False else float("nan")
        np.savez(os.path.join(a.out, f"L{a.layer}_w{w}.npz"), E=E.astype(np.float32))
        rep = residual_structure(E, topology=True, n_points=2000, n_boot=20)
        rows.append(dict(model=a.model, release=a.release, layer=a.layer, width=w, d_sae=int(sae.cfg.d_sae) if hasattr(sae.cfg, "d_sae") else -1, n_tokens=len(E),
                         resid_norm_mean=float(np.linalg.norm(E, axis=1).mean()), n_outliers=int(rep.n_outliers), structured_fraction=float(rep.structured_fraction),
                         spectral_test_valid=bool(rep.spectral_test_valid), tc_ratio=float(getattr(rep, "tc_ratio", np.nan)), verdict=rep.verdict,
                         wall_s=round(time.time() - t0), note="replication of Engels et al. 2410.14670 with the spectral statistic (NOVELTY_BOARD #4)"))
        pd.DataFrame(rows).to_csv(csv_p, index=False); print(rows[-1], flush=True); del sae; torch.cuda.empty_cache()
    df = pd.DataFrame(rows).sort_values("d_sae")
    if len(df) >= 3:
        k = np.polyfit(np.log(df.d_sae), np.log(df.structured_fraction.clip(1e-6)), 1)[0]
        json.dump(dict(slope_log_structured_fraction_vs_log_width=float(k), agrees_with_engels_no_shrink=bool(abs(k) < 0.1),
                       rows=len(df)), open(os.path.join(a.out, "K7_summary.json"), "w"), indent=1); print("slope", k)


if __name__ == "__main__":
    main()
