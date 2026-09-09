"""C4 — audited concept map at scale (HPCC): the C1/C2 pipeline (experiments/C_concepts/run_C1_C3.py) on Gemma-2-9B, one
mid layer, with a k-sparse dictionary fitted on the MLP-hidden (post-gate) activations — the coordinates the gauge group
acts on (a residual-stream Gemma Scope SAE sees no MLP-unit permutation, so it cannot be gauge-audited; stated, not
hidden). Certification rule (C2): a feature is certified iff its blind activation-profile match is >= 0.95 under every
exact renaming (``symmetry.llama``: MLP perms, head perms, V/O rotations, RoPE-commuting Q/K phases, RMSNorm rescales).
Seed-consistency of features is a *baseline* here, not the claim (NOVELTY_BOARD.md #7: 2606.12138).

Usage:  python c4_scale.py --model google/gemma-2-9b --layer 20 --n-seq 400 --n-renamings 10 --out results/c4 --resume
Needs: HF_TOKEN, >= 40 GB GPU (bf16 model + renamed copy: use --cpu-copy to keep the renamed copy on CPU), spaCy
en_core_web_sm (POS concepts), whiteboxtopo. Checkpoints: dictionary npz, per-renaming rows (CSV), C2 counts."""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
CC = os.path.join(ROOT, "experiments", "C_concepts"); sys.path.insert(0, CC); sys.path.insert(0, os.path.join(ROOT, "experiments", "K_darkmatter"))
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants.check import per_component_cv, INVARIANT_CV
from scipy.optimize import linear_sum_assignment


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/gemma-2-9b"); ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-seq", type=int, default=400); ap.add_argument("--n-renamings", type=int, default=10); ap.add_argument("--n-lat", type=int, default=16384)
    ap.add_argument("--out", default="results/c4"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--dtype", default="bfloat16"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    import run_C1_C3 as C, run_K1
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
    run_K1.N_SEQ, run_K1.CACHE = a.n_seq, a.out; toks = run_K1.load_tokens(tok); T_ = torch.tensor(toks, device=dev)
    # concept labels: 20 most frequent fine POS tags (spaCy), aligned to model tokens by character offsets (as C1)
    import spacy; nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"]); labels = []
    for row in toks:
        text = tok.decode(row[1:]); doc = nlp(text); offs = tok(text, return_offsets_mapping=True)["offset_mapping"]; tags = ["-"] * len(offs)
        for t in doc:
            for i, (x0, x1) in enumerate(offs):
                if x0 <= t.idx < x1 or (x0 < t.idx + len(t) <= x1): tags[i] = t.tag_
        labels.append((tags + ["-"] * (toks.shape[1] - 1))[:toks.shape[1] - 1])
    labels = np.array(labels).ravel(); top20 = pd.Series(labels[labels != "-"]).value_counts().index[:20].tolist(); Y = np.stack([(labels == t).astype(float) for t in top20], 1)
    blk = lambda m: m.model.layers[a.layer]
    def mlp_acts(m):
        acts = []; h = blk(m).mlp.down_proj.register_forward_pre_hook(lambda mod, i: acts.append(i[0][:, 1:].reshape(-1, i[0].shape[-1]).float().cpu().numpy()))
        with torch.no_grad():
            for i in range(0, len(T_), 4): m(T_[i:i + 4])
        h.remove(); return np.concatenate(acts)
    A0 = mlp_acts(model); print("MLP acts", A0.shape, flush=True)
    dp = os.path.join(a.out, f"ksparse_L{a.layer}.npz")
    dic = dict(np.load(dp)) if os.path.exists(dp) else C.ksparse_dict(A0, n_lat=a.n_lat)
    if not os.path.exists(dp): np.savez(dp, **dic)
    hold = np.arange(len(A0)) % 5 == 0
    def probes(Z):
        from sklearn.linear_model import Ridge
        out = []
        for j in range(Y.shape[1]):
            r = Ridge(alpha=10.0).fit(Z[~hold], Y[~hold, j]); p = r.predict(Z[hold]); y = Y[hold, j]; out.append(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        return np.array(out)
    base_Z = C.encode(dic, A0); base_probe = probes(base_Z); n_lat = dic["D"].shape[1]
    rows_p = os.path.join(a.out, "C4_audit_rows.csv"); rows = pd.read_csv(rows_p).to_dict("records") if (a.resume and os.path.exists(rows_p)) else []
    cert_p = os.path.join(a.out, "C4_cert_counts.npy"); cert = np.load(cert_p) if (a.resume and os.path.exists(cert_p)) else np.zeros(n_lat)
    done = {(r["kind"], r["r"]) for r in rows}
    for kind in ("perm", "scale", "rot", "combined"):
        rng = np.random.default_rng(hash(kind) % 1000)
        for r in range(a.n_renamings):
            g = S.random_element(model, rng, kind=kind, orthogonal=False)
            if (kind, r) in done: continue
            t0 = time.time(); mg = S.apply(model, g); A = mlp_acts(mg); p = g.perms.get(("mlp", a.layer)); dic_g = dict(dic)
            if p is not None: dic_g["W"] = dic["W"][p]; dic_g["mu"] = dic["mu"][p]; dic_g["D"] = dic["D"][:, p]
            Zg = C.encode(dic_g, A); corr = np.nan_to_num(np.corrcoef(base_Z[::20].T, Zg[::20].T)[:n_lat, n_lat:]); ri, ci = linear_sum_assignment(-corr); m = np.zeros(n_lat); m[ri] = corr[ri, ci]
            if kind == "combined": cert += (m >= 0.95); np.save(cert_p, cert)
            rows.append(dict(kind=kind, r=r, probe_cv=float(np.median(per_component_cv(np.stack([base_probe, probes(Zg)])))), feat_identity_exact=float(np.mean(np.abs(base_Z - Zg).max(0) < 1e-3)),
                             blind_match_mean=float(m.mean()), blind_certified_frac=float((m >= 0.95).mean()), wall_s=round(time.time() - t0)))
            pd.DataFrame(rows).to_csv(rows_p, index=False); print(rows[-1], flush=True); del mg; torch.cuda.empty_cache()
    alive = (base_Z > 0).mean(0) > 0; certified = cert == a.n_renamings
    pd.DataFrame(dict(feature=np.arange(n_lat), certified=certified, alive=alive, cert_count=cert)).to_csv(os.path.join(a.out, "C4_inventory.csv"), index=False)
    summ = dict(model=a.model, layer=a.layer, n_lat=int(n_lat), alive=int(alive.sum()), certified_fraction_alive=float(certified[alive].mean()),
                probe_verdict={k: ("invariant" if np.median([r["probe_cv"] for r in rows if r["kind"] == k]) < INVARIANT_CV else "artifact") for k in ("perm", "scale", "rot", "combined")})
    json.dump(summ, open(os.path.join(a.out, "C4_summary.json"), "w"), indent=1); print(summ)


if __name__ == "__main__":
    main()
