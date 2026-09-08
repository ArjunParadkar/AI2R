"""K6 — certified dark matter (SPEC §K6). The K2/K3 statistics through ``audit_quantities`` under GPT-2 renamings
(head perm, within-head rotation, MLP-unit perm, LN scalings; n = 20 per kind; layers 0..L).
(a) residual-stream anchor: e = x - SAE(x) at blocks.L.hook_resid_pre — no internal renaming acts on the residual
    stream, so its statistics must be invariant to the fp32 floor (sanity anchor, stated as such);
(b) non-vacuous part: MLP-hidden-space residual e_mlp = h - PCA_256(h) of block L's post-GELU activations, where
    MLP-unit permutations DO act on the coordinates: spectral/topological statistics expected invariant,
    unit-indexed quantities (per-unit residual variance, top-20 units) expected artifact -> invariant after gauge alignment.
Runs in ~/Whitebox/v2-programme/.venv-tl (SAELens) on the K1 token set. Writes results/K6_certification.csv."""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root (whiteboxtopo) for the TL venv
from whiteboxtopo import symmetry as S
from whiteboxtopo.invariants import audit_quantities
from whiteboxtopo.symmetry.group import inverse_on_units
from whiteboxtopo.darkmatter import mp_outliers
from whiteboxtopo.darkmatter.topo import tc_features
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); R = os.path.join(HERE, "results")
LAYERS = [3, 6, 9]; BATCH = 8; K_PCA = 256; dev = "cuda"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    from transformers import GPT2LMHeadModel
    from sae_lens import SAE
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    toks = np.load(os.path.join(CACHE, "tokens.npy")); T = torch.as_tensor(toks, device=dev)
    if a.smoke: T = T[:32]
    rows = []
    for L in LAYERS:
        t0 = time.time(); hook = f"blocks.{L}.hook_resid_pre"
        sae = SAE.from_pretrained("gpt2-small-res-jb", hook, device=dev); sae = sae[0] if isinstance(sae, tuple) else sae; sae.eval()

        def quantities(m, L=L, sae=sae):
            cap = []
            hk = m.transformer.h[L].mlp.act.register_forward_hook(lambda mod, i, o: cap.append(o.detach()))
            Es, Hs = [], []
            with torch.no_grad():
                for b in range(0, T.shape[0], BATCH):
                    out = m(T[b:b + BATCH], output_hidden_states=True)
                    x = out.hidden_states[L]; x = x - x.mean(-1, keepdim=True)     # TL center_writing_weights convention (K1)
                    e = x - sae.decode(sae.encode(x))
                    Es.append(e[:, 1:].reshape(-1, x.shape[-1]).float().cpu()); Hs.append(cap[-1][:, 1:].reshape(-1, cap[-1].shape[-1]).float().cpu()); cap.clear()
            hk.remove(); torch.cuda.empty_cache()
            E_np = torch.cat(Es).numpy().astype(np.float64); Hm = torch.cat(Hs).numpy().astype(np.float64)   # CPU: the GPU is shared (OOM 2026-09-08)
            Hc = Hm - Hm.mean(0); C = Hc.T @ Hc / len(Hc); ev, V = np.linalg.eigh(C); Vk = V[:, -K_PCA:]
            Em_np = Hc - (Hc @ Vk) @ Vk.T                                         # PCA-256 residual in MLP-hidden space
            del Es, Hs, Hm, Hc, C
            Q = {}
            sub = np.random.default_rng(0).choice(len(E_np), min(1000, len(E_np)), replace=False)   # fixed subsample: the same tokens under every renaming
            for name, A in (("resid", E_np), ("mlp", Em_np)):
                mp = mp_outliers(A, form="correlation")
                Q[f"{name}_n_outliers"] = np.array([mp.n_outliers], float); Q[f"{name}_structured_fraction"] = np.array([mp.structured_fraction])
                Q[f"{name}_top20_eigs"] = np.sort(np.sort(mp.eigenvalues)[::-1][:20])
                tr = tc_features(A[sub], n_components=20, n_shuffle=3, seed=0)
                Q[f"{name}_tc_ratio"] = np.array([tr["tc_ratio"]]); Q[f"{name}_b1"] = np.array([tr["b1"]], float)
            var_u = Em_np.var(0)
            Q["mlp_per_unit_resid_var"] = var_u
            Q["mlp_per_unit_resid_var_sorted"] = np.sort(var_u)
            Q["mlp_top20_units"] = np.sort(np.argsort(-var_u)[:20]).astype(float)
            return Q

        key = ("mlp", L)
        def topk_align(g, v, key=key):
            p = g.perms.get(key); return v if p is None else np.sort(np.asarray(p)[v.astype(int)]).astype(float)
        aligners = {"mlp_per_unit_resid_var": (lambda g, v, k=key: inverse_on_units(g, k, v, axis=0)), "mlp_top20_units": topk_align}
        res = audit_quantities(quantities, model, n=a.n, seed=L, kinds=("perm", "scale", "rot", "combined"), aligners=aligners,
                               orthogonal=False, layers=list(range(L + 1)))
        for q, r in res.items():
            row = r.row(); row.update(layer=L, space="resid" if q.startswith("resid") else "mlp",
                                      expected="invariant (anchor: residual stream not acted on)" if q.startswith("resid")
                                      else ("artifact -> invariant after gauge alignment" if q in aligners else "invariant"))
            rows.append(row); print(L, q, r, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(R, "K6_certification.csv"), index=False)
        print(f"layer {L} {time.time()-t0:.0f}s", flush=True); del sae; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
