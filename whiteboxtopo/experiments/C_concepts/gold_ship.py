"""C1–C3 — audited concept map on GPT-2 small layer 6 (SPEC §C1–C3). Run in the TL venv
(~/Whitebox/v2-programme/.venv-tl) for the residual SAE; the MLP-space dictionary is fitted here.
Writes results/C1_audit.csv, C2_inventory.csv, C3_geometry.csv, C3_mapper.csv."""
import os, sys, json, time, csv
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache"); os.makedirs(R, exist_ok=True)
K1 = os.path.join(HERE, "..", "K_darkmatter", "cache")
sys.path.insert(0, os.path.join(HERE, "..", "F_gauge"))
from whiteboxtopo import symmetry as S, topology as T
from whiteboxtopo.invariants.check import per_component_cv, INVARIANT_CV
from whiteboxtopo.symmetry.group import inverse_on_units
LAYER, N_LAT, K_SPARSE, N_TOK = 6, 4096, 32, 50000
dev = "cuda"


def ksparse_dict(A, n_lat=N_LAT, k=K_SPARSE, steps=3000, seed=0, log=print):
    """k-sparse autoencoder (Makhzani & Frey / Gao et al. 2024 TopK) on MLP-hidden activations A (n, d)."""
    torch.manual_seed(seed); n, d = A.shape; At = torch.tensor(A, device=dev); mu = At.mean(0); At = At - mu
    W = torch.nn.Parameter(torch.randn(d, n_lat, device=dev) / np.sqrt(d)); b = torch.nn.Parameter(torch.zeros(n_lat, device=dev))
    D = torch.nn.Parameter(W.detach().T.clone())
    opt = torch.optim.Adam([W, b, D], lr=1e-3)
    for s in range(steps):
        idx = torch.randint(0, n, (1024,), device=dev); x = At[idx]
        pre = x @ W + b; top = torch.topk(pre, k, dim=1); z = torch.zeros_like(pre).scatter_(1, top.indices, torch.relu(top.values))
        rec = z @ D; loss = ((rec - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): D.data = D.data / (D.data.norm(dim=1, keepdim=True) + 1e-8)
        if s % 500 == 0: log(f"  ksparse step {s} loss {loss.item():.4f}")
    with torch.no_grad():
        pre = At @ W + b; top = torch.topk(pre, k, dim=1); z = torch.zeros_like(pre).scatter_(1, top.indices, torch.relu(top.values))
        fvu = float(((z @ D - At) ** 2).sum() / (At ** 2).sum())
    return dict(W=W.detach().cpu().numpy(), b=b.detach().cpu().numpy(), D=D.detach().cpu().numpy(), mu=mu.cpu().numpy(), fvu=fvu)


def encode(dic, A):
    pre = (A - dic["mu"]) @ dic["W"] + dic["b"]; thr = -np.sort(-pre, axis=1)[:, K_SPARSE - 1:K_SPARSE]
    return np.where(pre >= thr, np.maximum(pre, 0), 0.0)


def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    toks = np.load(os.path.join(K1, "tokens.npy")); T_ = torch.tensor(toks, device=dev)
    # concepts: 20 most frequent fine POS tags via spaCy on the detokenised text (token-level labels)
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    labels = []
    for row in toks:
        text = tok.decode(row[1:]); doc = nlp(text)
        # align spaCy tokens to GPT-2 tokens by character offsets
        offs = tok(text, return_offsets_mapping=True)["offset_mapping"]; tags = ["-"] * len(offs)
        for tkn in doc:
            for i, (a, b) in enumerate(offs):
                if a <= tkn.idx < b or (a < tkn.idx + len(tkn) <= b): tags[i] = tkn.tag_
        tags = (tags + ["-"] * (toks.shape[1] - 1))[:toks.shape[1] - 1]; labels.append(tags)
    labels = np.array(labels).ravel(); top20 = pd.Series(labels[labels != "-"]).value_counts().index[:20].tolist()
    Y = np.stack([(labels == t).astype(float) for t in top20], 1)

    def mlp_acts(m):
        acts = []
        h = m.transformer.h[LAYER].mlp.act.register_forward_hook(lambda mod, i, o: acts.append(o[:, 1:].reshape(-1, o.shape[-1]).float().cpu().numpy()))
        with torch.no_grad():
            for i in range(0, len(T_), 16): m(T_[i:i + 16])
        h.remove(); return np.concatenate(acts)

    A0 = mlp_acts(model); print("MLP acts", A0.shape, flush=True)
    dp = os.path.join(CACHE, "ksparse_L6.npz")
    if os.path.exists(dp): dic = dict(np.load(dp))
    else: dic = ksparse_dict(A0); np.savez(dp, **dic)
    print("dictionary FVU", float(dic["fvu"]), flush=True)
    hold = np.arange(len(A0)) % 5 == 0
    def probes(Z):
        from sklearn.linear_model import Ridge
        out = []
        for j in range(Y.shape[1]):
            r = Ridge(alpha=10.0).fit(Z[~hold], Y[~hold, j]); p = r.predict(Z[hold]); y = Y[hold, j]
            out.append(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        return np.array(out)

    # ---- C1: audit under renamings with consistent dictionary transform (MLP-unit perms: permute W rows / D columns)
    base_Z = encode(dic, A0); base_geom = dic["D"] @ dic["D"].T; base_probe = probes(base_Z)
    samples = {"perm": [], "scale": [], "rot": [], "combined": []}; cos_aligned = {k: [] for k in samples}
    t0 = time.time()
    for kind in samples:
        rng = np.random.default_rng(hash(kind) % 1000)
        for r in range(20):
            g = S.random_element(model, rng, kind=kind, orthogonal=False); mg = S.apply(model, g); A = mlp_acts(mg)
            p = g.perms.get(("mlp", LAYER))
            dic_g = dict(dic); 
            if p is not None:   # transformed unit j = base unit p[j]: rows of W / mu / columns of D follow
                dic_g["W"] = dic["W"][p]; dic_g["mu"] = dic["mu"][p]; dic_g["D"] = dic["D"][:, p]
            Zg = encode(dic_g, A)
            # blind: features matched by decoder-direction cosine after undoing NOTHING (columns of D are in unit coords, so
            # under a unit permutation the direction vectors are permuted entries; blind Hungarian on the *activation profile* instead)
            act_corr = np.corrcoef(base_Z[::20].T, Zg[::20].T)[:N_LAT, N_LAT:]
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(-np.nan_to_num(act_corr)); blind_match = np.nan_to_num(act_corr)[ri, ci]
            samples[kind].append(dict(probe_r2=probes(Zg), geom=(dic_g["D"] @ dic_g["D"].T)[::64, ::64].ravel(), feat_identity_exact=np.mean(np.abs(base_Z - Zg).max(0) < 1e-3),
                                      blind_match_mean=float(blind_match.mean()), blind_certified_frac=float((blind_match >= 0.95).mean()), max_dlogit=float("nan")))
            print(kind, r, f"{time.time()-t0:.0f}s", samples[kind][-1]["feat_identity_exact"], samples[kind][-1]["blind_certified_frac"], flush=True)
            del mg; torch.cuda.empty_cache()
    rows = []
    for q in ("probe_r2", "geom"):
        for kind, ss in samples.items():
            V = np.stack([base_probe if q == "probe_r2" else base_geom[::64, ::64].ravel()] + [s[q] for s in ss]); cv = per_component_cv(V)
            rows.append(dict(quantity=q, kind=kind, median_cv=float(np.median(cv)), frac_invariant=float(np.mean(cv < INVARIANT_CV)), verdict="invariant" if np.median(cv) < INVARIANT_CV else "artifact"))
    for kind, ss in samples.items():
        rows.append(dict(quantity="feature_identity_exact_frac", kind=kind, median_cv=float(np.mean([s["feat_identity_exact"] for s in ss])), frac_invariant=np.nan, verdict=""))
        rows.append(dict(quantity="blind_certified_frac", kind=kind, median_cv=float(np.mean([s["blind_certified_frac"] for s in ss])), frac_invariant=np.nan, verdict=""))
    pd.DataFrame(rows).to_csv(os.path.join(R, "gold_ship_audit.csv"), index=False); print(pd.DataFrame(rows).to_string(), flush=True)
    # ---- C2 certified inventory (blind, across all samples): feature certified iff matched corr >= 0.95 in every renaming
    cert_counts = np.zeros(N_LAT)
    n_samples = 0
    for kind in ("combined",):
        rng = np.random.default_rng(999)
        for r in range(10):
            g = S.random_element(model, rng, kind=kind, orthogonal=False); mg = S.apply(model, g); A = mlp_acts(mg)
            p = g.perms.get(("mlp", LAYER)); dic_g = dict(dic); dic_g["W"] = dic["W"][p]; dic_g["mu"] = dic["mu"][p]; dic_g["D"] = dic["D"][:, p]
            Zg = encode(dic_g, A); act_corr = np.nan_to_num(np.corrcoef(base_Z[::20].T, Zg[::20].T)[:N_LAT, N_LAT:])
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(-act_corr); m = np.zeros(N_LAT); m[ri] = act_corr[ri, ci]; cert_counts += (m >= 0.95); n_samples += 1
            del mg; torch.cuda.empty_cache()
    certified = cert_counts == n_samples; alive = (base_Z > 0).mean(0) > 0
    pd.DataFrame(dict(feature=np.arange(N_LAT), certified=certified, alive=alive, cert_count=cert_counts)).to_csv(os.path.join(R, "gold_ship_inventory.csv"), index=False)
    print("C2 certified fraction (alive features):", float(certified[alive].mean()), "alive", int(alive.sum()), flush=True)
    # ---- C3 geometry + Mapper on certified vs uncertified lenses
    Dc = dic["D"][certified & alive]; ang = Dc @ Dc.T
    dg = T.rips(Dc[:1500] if len(Dc) > 1500 else Dc, maxdim=1, backend="ripser")
    geo = dict(n_certified=int((certified & alive).sum()), mean_abs_cos=float(np.abs(ang[np.triu_indices(len(ang), 1)]).mean()) if len(ang) > 1 else np.nan,
               n_bars_H1=int(len(dg.dgms[1])), total_persistence_H1=float(np.sum((np.asarray(dg.dgms[1]).reshape(-1, 2)[:, 1] - np.asarray(dg.dgms[1]).reshape(-1, 2)[:, 0]) ** 2)) if len(dg.dgms[1]) else 0.0)
    pd.DataFrame([geo]).to_csv(os.path.join(R, "gold_ship_geometry.csv"), index=False)
    sub = np.random.default_rng(3).choice(len(A0), 3000, replace=False); rows = []
    cidx = np.flatnonzero(certified & alive); uidx = np.flatnonzero(~certified & alive)
    for lens_name, idx in (("certified", cidx), ("uncertified", uidx)):
        if len(idx) < 2: continue
        top2 = idx[np.argsort(-base_Z[:, idx].mean(0))[:2]]; lens = base_Z[sub][:, top2]
        mp = T.mapper(A0[sub], lens, n_cubes=8, overlap=0.3); rows.append(dict(lens=lens_name, n_nodes=mp.n_nodes, n_edges=mp.n_edges, n_components=mp.n_components, cycle_rank=mp.cycle_rank))
    pd.DataFrame(rows).to_csv(os.path.join(R, "gold_ship_mapper.csv"), index=False); print(geo, rows, flush=True); print("done")


if __name__ == "__main__":
    main()
