"""B2 (bias direction: real geometry?), B3 (power curve), B4 (separability rank) — SPEC §B2–B4.
GPT-2 small pretrained (HF, fp32). Writes results/B2_direction.csv, B2_kill.json, B3_power.csv, B4_rank.csv."""
import os, sys, json, time
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blib import build_corpus, build_corpus_unique, encode, MALE_OCC, FEMALE_OCC
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import audit_quantities
from whiteboxtopo.symmetry.group import inverse_on_units
from whiteboxtopo.core.persistence import compute_diagrams
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(R, exist_ok=True)
BLOCKS = [1, 2, 3, 6]; dev = "cuda"
B3_NS = (25, 50, 100, 200); B3_NS_REAL = (25, 50, 100, 200, 400); B3_PROCS = 8; B3_NPERM = 100   # AMENDMENTS #1, #2
_B3 = {}


def pooled(model, ids, mask, block):
    """Mean-pooled residual (after block), MLP-hidden (post-GELU) and per-head outputs z of `block`."""
    cap = {}
    h1 = model.transformer.h[block].mlp.act.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp", o.detach()))
    h2 = model.transformer.h[block].attn.register_forward_pre_hook(lambda m, i: cap.__setitem__("ln1", i[0].detach()))
    with torch.no_grad():
        out = model(ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
    h1.remove(); h2.remove()
    mk = mask.to(dev).float()[:, :, None]
    def mp(x): return ((x * mk).sum(1) / mk.sum(1)).float().cpu().numpy()
    resid = mp(out.hidden_states[block + 1]); mlp = mp(cap["mlp"])
    d = model.config.n_embd; H = model.config.n_head; dh = d // H; x = cap["ln1"]
    with torch.no_grad():
        q, k, v = model.transformer.h[block].attn.c_attn(x).split(d, dim=2); B, T, _ = q.shape
        q = q.view(B, T, H, dh).transpose(1, 2); k = k.view(B, T, H, dh).transpose(1, 2); v = v.view(B, T, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None, None] & mask.to(dev).bool()[:, None, None, :]
        att = att.masked_fill(~causal, float("-inf")).softmax(-1); z = (att @ v).permute(0, 2, 1, 3).reshape(B, T, H * dh)
    heads = mp(z).reshape(-1, H, dh)
    return resid, mlp, heads


def unit(v): return v / (np.linalg.norm(v) + 1e-12)


def _biastop(A, B):
    from persim import wasserstein
    mu, P20 = _B3["mu"], _B3["P20"]
    da = compute_diagrams((A - mu) @ P20, maxdim=1, backend="ripser").dgms[1]; db = compute_diagrams((B - mu) @ P20, maxdim=1, backend="ripser").dgms[1]
    return wasserstein(np.asarray(da).reshape(-1, 2), np.asarray(db).reshape(-1, 2))


def _perm_test(A, B, rng, n_perm=B3_NPERM):
    stat = _biastop(A, B); Z = np.vstack([A, B]); n = len(A); c = 0
    for i in range(n_perm):
        p = rng.permutation(len(Z)); c += _biastop(Z[p[:n]], Z[p[n:]]) >= stat
    return (c + 1) / (n_perm + 1)


def _b3_trial(args):
    """One B3 trial -> 1 if the permutation test rejects at alpha = 0.05. variant: shift (pre-registered translation,
    == null by translation invariance), stretch (factor 1+delta along g; AMENDMENTS #1), real_MF (male vs female clouds)."""
    variant, n, delta, seed = args
    import torch; torch.set_num_threads(1)
    M, F, g, sd = _B3["M"], _B3["F"], _B3["g"], _B3["sd"]
    rng = np.random.default_rng([seed, n, {"shift": 0, "stretch": 1, "real_MF": 2}[variant]])
    if variant == "real_MF":
        A = M[rng.choice(len(M), n, replace=False)]; B = F[rng.choice(len(F), n, replace=False)]
    else:
        idx = rng.choice(len(M), 2 * n, replace=False); A, B = M[idx[:n]], M[idx[n:]]
        if variant == "shift": B = B + float(delta) * sd * g
        else: B = B + float(delta) * ((B - B.mean(0)) @ g)[:, None] * g[None, :]
    return int(_perm_test(A, B, rng) < 0.05)


def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    male, female = build_corpus(0, 100); m_ids, m_mask = encode(tok, male); f_ids, f_mask = encode(tok, female)
    rows, kill = [], {}
    b2_kill_p = os.path.join(R, "B2_kill.json")
    b2_done = os.path.exists(b2_kill_p) and all(f"block{b}" in json.load(open(b2_kill_p)) for b in BLOCKS)
    if b2_done: print("B2 complete on disk (B2_direction.csv, B2_kill.json) — skipped on resume", flush=True)
    for block in ([] if b2_done else BLOCKS):
        def quantities(m):
            rm, mm, hm = pooled(m, m_ids, m_mask, block); rf, mf, hf = pooled(m, f_ids, f_mask, block)
            Q = {"g_resid": unit(rm.mean(0) - rf.mean(0)), "g_resid_norm": np.array([np.linalg.norm(rm.mean(0) - rf.mean(0))]),
                 "g_mlp": unit(mm.mean(0) - mf.mean(0)), "g_mlp_norm": np.array([np.linalg.norm(mm.mean(0) - mf.mean(0))]),
                 "g_heads": (hm.mean(0) - hf.mean(0)) / (np.linalg.norm(hm.mean(0) - hf.mean(0)) + 1e-12),
                 "g_heads_per_head_norm": np.linalg.norm(hm.mean(0) - hf.mean(0), axis=1)}
            return Q
        key = ("mlp", block)
        def head_align(g, v):
            p = g.perms.get(("head", block)); rots = g.rots.get(("rot", block))
            out = v.copy()
            if rots is not None:   # z' = z R per head (V rotated by R) -> undo: z = z' R^{-1}
                for h, Rm in enumerate(rots): out[h] = out[h] @ np.linalg.inv(np.asarray(Rm))
            if p is not None:
                inv = np.empty_like(p); inv[p] = np.arange(len(p)); out = out[inv]
            return out
        def head_norm_align(g, v):
            p = g.perms.get(("head", block))
            if p is None: return v
            inv = np.empty_like(p); inv[p] = np.arange(len(p)); return v[inv]
        aligners = {"g_mlp": lambda g, v, k=key: inverse_on_units(g, k, v, axis=0), "g_heads": head_align, "g_heads_per_head_norm": head_norm_align}
        t0 = time.time()
        res = audit_quantities(quantities, model, n=20, kinds=("perm", "scale", "rot", "combined"), aligners=aligners, orthogonal=False)
        for q, r in res.items():
            row = r.row(); row.update(block=block)
            # direction stability: cosine of each sample with the base, raw and gauge-aligned
            if q in ("g_resid", "g_mlp", "g_heads"):
                for kind, Vs in r.samples.items():
                    b = Vs[0].ravel(); row[f"cos_raw_{kind}"] = float(np.mean([np.dot(v.ravel(), b) / (np.linalg.norm(v) * np.linalg.norm(b)) for v in Vs[1:]]))
                if q in aligners:
                    # rebuild aligned samples from stored raw ones is not possible here (g not stored); report aligned CV instead
                    pass
            rows.append(row); print(block, q, r, flush=True)
        print(f"block {block} audit {time.time()-t0:.0f}s", flush=True)
        # ---- B2 kill: random-split directions vs M/F-split directions from disjoint halves
        rm, mm, hm = pooled(model, m_ids, m_mask, block); rf, mf, hf = pooled(model, f_ids, f_mask, block)
        allX = np.vstack([rm, rf]); rng = np.random.default_rng(block); cos_rand, cos_mf = [], []
        for t in range(20):
            p = rng.permutation(len(allX)); a, b = allX[p[:100]], allX[p[100:]]
            p2 = rng.permutation(len(allX)); a2, b2 = allX[p2[:100]], allX[p2[100:]]
            cos_rand.append(float(np.dot(unit(a.mean(0) - b.mean(0)), unit(a2.mean(0) - b2.mean(0)))))
            pm, pf = rng.permutation(100), rng.permutation(100)
            g1 = unit(rm[pm[:50]].mean(0) - rf[pf[:50]].mean(0)); g2 = unit(rm[pm[50:]].mean(0) - rf[pf[50:]].mean(0)); cos_mf.append(float(np.dot(g1, g2)))
        kill[f"block{block}"] = dict(cos_random_split_mean=float(np.mean(cos_rand)), cos_random_split_absmean=float(np.mean(np.abs(cos_rand))),
                                     cos_mf_disjoint_halves_mean=float(np.mean(cos_mf)), pass_kill=bool(np.mean(np.abs(cos_rand)) < 0.5 and np.mean(cos_mf) > 0.9))
        print("kill", block, kill[f"block{block}"], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(R, "B2_direction.csv"), index=False); json.dump(kill, open(os.path.join(R, "B2_kill.json"), "w"), indent=1)
    # ---- B3 power curve (block 6 residual clouds, PCA-20, BiasTop = W2 between H1 diagrams) — see AMENDMENTS #1:
    # the pre-registered planted divergence (a translation of the cloud by δ·sd·g) is invisible to a Vietoris–Rips
    # diagram by construction, so the shift cells measure the false-positive rate; a post-hoc *stretch* variant
    # (factor 1+δ along g) is added as the calibrated power curve; trials are parallel and the CSV is resumable.
    from multiprocessing import Pool
    # AMENDMENTS #2: unique-sentence pools (480 per group from 40 frames x 12 occupations), encoded in batches
    mu_s, fu_s = build_corpus_unique(0)
    def pool_all(sents):
        out = []
        for i in range(0, len(sents), 120):
            ids, mk = encode(tok, sents[i:i + 120]); out.append(pooled(model, ids, mk, 6)[0])
        return np.vstack(out)
    M_all, F_all = pool_all(mu_s), pool_all(fu_s)
    print(f"B3 pools: {len(M_all)} unique male, {len(F_all)} unique female sentences", flush=True)
    assert 2 * max(B3_NS) <= len(M_all) and max(B3_NS_REAL) <= min(len(M_all), len(F_all))
    allX = np.vstack([M_all, F_all]); mu = allX.mean(0); U, s, Vt = np.linalg.svd(allX - mu, full_matrices=False); P20 = Vt[:20].T
    g = unit(M_all.mean(0) - F_all.mean(0)); sd_pool = float(allX.std(0).mean())
    _B3.update(M=M_all, F=F_all, mu=mu, P20=P20, g=g, sd=sd_pool)
    b3_csv = os.path.join(R, "B3_power.csv")
    prow = pd.read_csv(b3_csv).to_dict("records") if os.path.exists(b3_csv) else []
    for r in prow:   # rows from the 2026-09-07 run: duplicated v1 corpus (60 distinct sentences), translation variant
        if not isinstance(r.get("variant"), str): r["variant"] = "shift_v1corpus"
    have = {(r["variant"], int(r["n"]), str(r["delta"])) for r in prow}
    cells = [("shift", n, 0.0) for n in B3_NS] + [("stretch", n, d) for n in B3_NS for d in (0.25, 0.5, 1.0)] + [("real_MF", n, "real_MF") for n in B3_NS_REAL]
    with Pool(B3_PROCS) as pool:
        for variant, n, delta in cells:
            if (variant, n, str(delta)) in have: continue
            trials = 50 if n <= 200 else 20; t0 = time.time()
            rej = sum(pool.map(_b3_trial, [(variant, n, delta, seed) for seed in range(trials)]))
            prow.append(dict(n=n, delta=delta, trials=trials, power=rej / trials, variant=variant)); print(prow[-1], f"{time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(prow).to_csv(b3_csv, index=False)
    # power-0.8 crossings (linear interpolation in n) and FPR summary
    df = pd.DataFrame(prow); summ = []
    for (variant, delta), grp in df.groupby(["variant", "delta"]):
        grp = grp.sort_values("n"); ns, pw = grp.n.values.astype(float), grp.power.values.astype(float); n80 = np.nan
        for a, b in zip(range(len(ns) - 1), range(1, len(ns))):
            if pw[a] < 0.8 <= pw[b]: n80 = ns[a] + (0.8 - pw[a]) * (ns[b] - ns[a]) / (pw[b] - pw[a]); break
        if pw[0] >= 0.8: n80 = ns[0]
        summ.append(dict(variant=variant, delta=delta, n_for_power_0_8=n80, max_power=pw.max(), fpr_ok=bool(pw.max() <= 0.07) if variant == "shift" and str(delta) == "0.0" else ""))
    pd.DataFrame(summ).to_csv(os.path.join(R, "B3_summary.csv"), index=False); print(summ, flush=True)
    # ---- B4 separability rank: participation ratio of the between/within discriminative spectrum
    def pr_spectrum(X, labels, k=50):
        Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False); Z = Xc @ Vt[:k].T
        groups = np.unique(labels); mu = Z.mean(0); Sb = sum((labels == g).sum() * np.outer(Z[labels == g].mean(0) - mu, Z[labels == g].mean(0) - mu) for g in groups)
        Sw = sum(np.cov(Z[labels == g].T) * ((labels == g).sum() - 1) for g in groups) / (len(Z) - len(groups))
        ev = np.linalg.eigvalsh(np.linalg.inv(Sw + 1e-6 * np.eye(k)) @ Sb); ev = np.clip(ev, 0, None)
        return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.sum() > 0 else 0.0
    b4 = [dict(attribute="gender", groups=2, participation_ratio=pr_spectrum(allX, np.r_[np.zeros(len(M_all)), np.ones(len(F_all))]), v1_erasure="succeeded (70.8 % at +1.9 % ppl)")]
    # COMPAS race and toxicity: v1 data files
    try:
        comp = pd.read_csv(os.path.expanduser("~/Whitebox/v1/experiments/data/compas.csv")).head(1200)
        texts = [f"The defendant is a {r.age}-year-old {r.sex}. Prior offenses: {r.priors_count}." for r in comp.itertuples()]
        ids, mk = encode(tok, texts, 48); X = np.vstack([pooled(model, ids[i:i + 64], mk[i:i + 64], 6)[0] for i in range(0, len(ids), 64)])
        lab = comp.race.astype("category").cat.codes.values
        b4.append(dict(attribute="COMPAS_race", groups=int(len(np.unique(lab))), participation_ratio=pr_spectrum(X, lab), v1_erasure="failed"))
    except Exception as e:
        b4.append(dict(attribute="COMPAS_race", groups=0, participation_ratio=np.nan, v1_erasure=f"data error: {e!r}"[:120]))
    try:
        toxp = os.path.expanduser("~/Whitebox/v1/experiments/data/toxicity.csv"); tx = pd.read_csv(toxp).head(1200)
        col = [c for c in tx.columns if "tox" in c.lower()][0]; tcol = [c for c in tx.columns if tx[c].dtype == object][0]
        ids, mk = encode(tok, tx[tcol].astype(str).tolist(), 48); X = np.vstack([pooled(model, ids[i:i + 64], mk[i:i + 64], 6)[0] for i in range(0, len(ids), 64)])
        lab = pd.qcut(tx[col], 4, labels=False, duplicates="drop").values
        b4.append(dict(attribute="toxicity", groups=int(len(np.unique(lab))), participation_ratio=pr_spectrum(X, lab), v1_erasure="failed"))
    except Exception as e:
        b4.append(dict(attribute="toxicity", groups=0, participation_ratio=np.nan, v1_erasure=f"data error: {e!r}"[:120]))
    pd.DataFrame(b4).to_csv(os.path.join(R, "B4_rank.csv"), index=False); print(b4)
    print("done")


if __name__ == "__main__":
    main()
