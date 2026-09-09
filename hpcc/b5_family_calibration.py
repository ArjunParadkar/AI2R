"""B5 — BiasTop per-family calibration (HPCC): the B3 power curve (false-positive rate at the Rips-invariant *shift*
cells, power at the *stretch* cells; B AMENDMENTS #1/#2) on GPT-2 small, Pythia-410M and Gemma-2-2B, with the stated
n requirement per family = smallest n per group with power >= 0.8 at stretch delta = 0.5 (and the FPR at that n).

Usage:  python b5_family_calibration.py --models gpt2 EleutherAI/pythia-410m google/gemma-2-2b --out results/b5 --resume
Needs: HF_TOKEN for Gemma; whiteboxtopo; experiments/B_bias (blib corpus, run_B2_B4._biastop/_perm_test/_b3_trial).
Checkpoint: one CSV row per (model, block, cell, n, trial); --resume skips finished (model, cell, n) groups."""
import os, sys, json, time, argparse, itertools
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
BB = os.path.join(ROOT, "experiments", "B_bias"); sys.path.insert(0, BB)
from blib import build_corpus_unique, encode
import run_B2_B4 as B


def pooled_resid(model, tok, sents, layer, dev, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(sents), bs):
            enc = tok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=40).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
            m = enc["attention_mask"].unsqueeze(-1).float(); out.append(((hs * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=["gpt2", "EleutherAI/pythia-410m", "google/gemma-2-2b"])
    ap.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200, 400]); ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--trials", type=int, default=20); ap.add_argument("--out", default="results/b5"); ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    csv_p = os.path.join(a.out, "B5_power.csv"); rows = pd.read_csv(csv_p).to_dict("records") if (a.resume and os.path.exists(csv_p)) else []
    done = {(r["model"], r["cell"], r["delta"], r["n"]) for r in rows}
    male, female = build_corpus_unique(0)
    for name in a.models:
        tok = AutoTokenizer.from_pretrained(name); tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
        n_layers = model.config.num_hidden_layers; layer = n_layers // 2      # block-mid, per family
        M = pooled_resid(model, tok, male, layer, dev); F = pooled_resid(model, tok, female, layer, dev)
        allX = np.vstack([M, F]); g = B.unit(M.mean(0) - F.mean(0))
        for cell, delta, n in itertools.product(("shift", "stretch"), a.deltas, a.ns):
            if (name, cell, delta, n) in done or n > min(len(M), len(F)): continue
            t0 = time.time(); ps = []
            for t in range(a.trials):
                rng = np.random.default_rng(1000 * t + n); p = rng.permutation(len(allX)); A_, B_ = allX[p[:n]].copy(), allX[p[n:2 * n]].copy()
                proj = (B_ - B_.mean(0)) @ g
                if cell == "shift": B_ = B_ + delta * allX.std() * g[None, :]                                  # Rips-invariant translation: FPR cell
                else: B_ = B_ + delta * proj[:, None] * g[None, :]                                                # stretch along g: calibrated power cell
                ps.append(B._perm_test(A_, B_, rng)[1] if hasattr(B, "_perm_test") else np.nan)
            ps = np.array(ps, dtype=float); rej = float(np.mean(ps <= 0.05))
            rows.append(dict(model=name, layer=layer, cell=cell, delta=delta, n=n, trials=a.trials, reject_rate=rej,
                             reading=("false_positive_rate" if (cell == "shift" or delta == 0) else "power"), wall_s=round(time.time() - t0)))
            pd.DataFrame(rows).to_csv(csv_p, index=False); print(rows[-1], flush=True)
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows); summ = []
    for name, d in df.groupby("model"):
        pw = d[(d.cell == "stretch") & (d.delta == 0.5)].sort_values("n"); ok = pw[pw.reject_rate >= 0.8]
        n_req = int(ok.n.min()) if len(ok) else None
        fpr = d[(d.cell == "shift") | (d.delta == 0)]
        summ.append(dict(model=name, n_required_power80_stretch05=n_req, fpr_max=float(fpr.reject_rate.max()) if len(fpr) else None,
                         note="shift cells are Rips-invariant by construction (B AMENDMENTS #1): their reject rate is the FPR, not power"))
    json.dump(summ, open(os.path.join(a.out, "B5_requirements.json"), "w"), indent=1); print(summ)


if __name__ == "__main__":
    main()
