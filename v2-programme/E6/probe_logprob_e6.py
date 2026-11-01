"""SPEC amendment 3: Test 3a with the log-prob delta target (softmax-invariant)."""
import os, sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6lib.probe import RidgeAccumulator, R2Evaluator
from e6lib.rmt import feature_shuffle
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); RES = os.path.join(HERE, "results")
N_SEQ, SEQ_LEN, DB = 512, 128, 8
from transformer_lens import HookedTransformer
from sae_lens import SAE
dev = "cuda"; t0 = time.time()
model = HookedTransformer.from_pretrained_no_processing("gpt2-small", device=dev, center_writing_weights=True)
tt = torch.as_tensor(np.load(os.path.join(CACHE, "tokens.npy"))); hold = np.arange(N_SEQ) % 5 == 0; V = model.cfg.d_vocab
rows = []
for L in (2, 6, 10):
    hook = f"blocks.{L}.hook_resid_pre"; sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=dev); sae = sae[0] if isinstance(sae, tuple) else sae
    z = np.load(os.path.join(CACHE, f"L{L}.npz")); X, E, sid = z["X"], z["E"], z["seq"]
    mu_e, sd_e = E.mean(0), E.std(0) + 1e-8; mu_x, sd_x = X.mean(0), X.std(0) + 1e-8
    tr_idx = np.flatnonzero(~hold[sid]); Esh = E.copy(); Esh[tr_idx] = feature_shuffle(E[tr_idx], seed=L)
    accs = {k: RidgeAccumulator(768, V, 1.0, device=dev, dtype=torch.float32) for k in ("E", "Eshuf", "X")}
    def batches(which):
        with torch.no_grad():
            for b in range(0, N_SEQ, DB):
                sel = np.arange(b, min(b + DB, N_SEQ)); sel = sel[hold[sel] == which]
                if len(sel) == 0: continue
                ids = tt[sel].to(dev)
                clean = torch.log_softmax(model(ids), -1); pat = torch.log_softmax(model.run_with_hooks(ids, fwd_hooks=[(hook, lambda a, hook=None: sae.decode(sae.encode(a)))]), -1)
                d = (clean - pat)[:, 1:].reshape(-1, V).float()
                yield np.concatenate([np.arange(s * (SEQ_LEN - 1), (s + 1) * (SEQ_LEN - 1)) for s in sel]), d
    feats = lambda ti: {"E": torch.as_tensor((E[ti] - mu_e) / sd_e), "Eshuf": torch.as_tensor((Esh[ti] - mu_e) / sd_e), "X": torch.as_tensor((X[ti] - mu_x) / sd_x)}
    for ti, d in batches(False):
        f = feats(ti); [accs[k].add(f[k], d) for k in accs]
    for a in accs.values(): a.fit()
    evs = {k: R2Evaluator(accs[k].train_mean()) for k in accs}; norms = []
    for ti, d in batches(True):
        f = feats(ti); [evs[k].add(d, accs[k].predict(f[k])) for k in accs]; norms.append(d.norm(dim=1).cpu().numpy())
    r = {k: evs[k].r2 for k in evs}; print(L, r, float(np.concatenate(norms).mean()), flush=True)
    rows.append(dict(layer=L, target="logprob_delta", r2_E=r["E"], r2_Eshuf=r["Eshuf"], r2_X=r["X"], mean_delta_norm=float(np.concatenate(norms).mean()), elapsed_s=time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "e6_probe_logprob.csv"), index=False)
    del accs, sae; torch.cuda.empty_cache()
