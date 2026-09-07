"""F7 — transformer extension (SPEC §F7). Trains 1- and 2-layer toy transformers on XOR3, MAJ3,
PARITY4, THR2 (seeds 0-9), enumerates all node subsets (heads + MLP units) under zero and mean
ablation, keeps minimal valid circuits, hashes gauge-canonical keys, and writes
results/orbits_transformer.csv (per specimen) + results/orbits_transformer_pooled.csv + F7_status.csv."""
import os, sys, csv, json, time
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from flib.f7 import train_toy, make_tokens, enumerate_valid_tf, canonical_key_tf, node_layout
from flib.minimal import minimal_ids_fast
from whiteboxtopo import symmetry as S

R = os.path.join(os.path.dirname(__file__), "results")
FUNCS, LAYERS, SEEDS = ("XOR3", "MAJ3", "PARITY4", "THR2"), (1, 2), range(10)
CAP = 1800


def main():
    torch.manual_seed(0)
    sp = os.path.join(R, "F7_status.csv"); done = set()
    if os.path.exists(sp): done = {r["specimen"] for r in csv.DictReader(open(sp))}
    ep = os.path.join(R, "F7_explanations.csv")
    for fn in FUNCS:
        for L in LAYERS:
            for s in (SEEDS if L == 1 else range(5)):   # AMENDMENTS #5
                spec = f"{fn}_L{L}_s{s}"
                if spec in done: continue
                t0 = time.time()
                model, conv, steps = train_toy(fn, L, s)
                tok, y = make_tokens(fn)
                st = dict(specimen=spec, function=fn, layers=L, seed=s, converged=int(conv), steps=steps, n_nodes=node_layout(model)[1])
                # identity check of the extended group on this model (F8-style, 30 combined renamings)
                rng = np.random.default_rng(s); worst = 0.0
                for r in range(30):
                    g = S.random_element(model, rng, kind="combined", orthogonal=(r % 2 == 0))
                    worst = max(worst, S.output_deviation(model, g, tok))
                st["max_dev_combined"] = worst
                rows = []
                if conv:
                    for scheme in ("zero", "mean"):
                        valid = enumerate_valid_tf(model, tok, y, scheme, time_cap=CAP)
                        if valid is None:
                            st[f"enumerated_{scheme}"] = 0; continue
                        mins = minimal_ids_fast(valid)
                        st[f"enumerated_{scheme}"] = 1; st[f"n_valid_{scheme}"] = int(len(valid)); st[f"n_minimal_{scheme}"] = int(len(mins))
                        # gauge-consistency check of the keys (kill-b analogue): keys of g.model must match
                        g = S.random_element(model, np.random.default_rng(100 + s), kind="combined")
                        mg = S.apply(model, g)
                        per, n_nodes = node_layout(model)
                        # image of each circuit id under the head/MLP permutations of g
                        def image(mid):
                            out = 0
                            for l in range(model.L):
                                ph = g.perms.get(("head", l), np.arange(model.H)); pm = g.perms.get(("mlp", l), np.arange(model.m))
                                for j in range(model.H):
                                    if (mid >> (l * per + int(ph[j]))) & 1: out |= 1 << (l * per + j)
                                for j in range(model.m):
                                    if (mid >> (l * per + model.H + int(pm[j]))) & 1: out |= 1 << (l * per + model.H + j)
                            return out
                        agree = 0
                        for mid in mins:
                            k0, parts = canonical_key_tf(model, int(mid), 1e-6); k1, _ = canonical_key_tf(mg, image(int(mid)), 1e-6)
                            agree += int(k0 == k1)
                            rows.append(dict(specimen=spec, criterion=scheme, circuit_id=int(mid), n_nodes=bin(int(mid)).count("1"),
                                             n_heads=sum(p[0] for p in parts), n_units=sum(p[1] for p in parts), key6=k0,
                                             key4=canonical_key_tf(model, int(mid), 1e-4)[0]))
                        st[f"key_gauge_agree_{scheme}"] = f"{agree}/{len(mins)}"
                st["wall"] = round(time.time() - t0, 1)
                for path, fields, rr in ((sp, list(st), [st]), (ep, ["specimen", "criterion", "circuit_id", "n_nodes", "n_heads", "n_units", "key6", "key4"], rows)):
                    if not rr: continue
                    new = not os.path.exists(path)
                    with open(path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                        if new: w.writeheader()
                        for r in rr: w.writerow(r)
                print(st, flush=True)
    # aggregate
    E = pd.read_csv(ep, dtype={"key6": str, "key4": str}); E["cell"] = E.specimen.str.replace(r"_s\d+$", "", regex=True)
    per_spec = E.groupby(["specimen", "criterion"]).agg(N=("circuit_id", "size"), K_within=("key6", "nunique")).reset_index()
    per_spec["ratio_within"] = per_spec.K_within / per_spec.N
    per_spec.to_csv(os.path.join(R, "orbits_transformer.csv"), index=False)
    pooled = E.groupby(["cell", "criterion"]).agg(N=("circuit_id", "size"), K_pooled=("key6", "nunique"), K_pooled_eps4=("key4", "nunique"),
                                                  n_specimens=("specimen", "nunique")).reset_index()
    pooled["ratio"] = pooled.K_pooled / pooled.N
    pooled.to_csv(os.path.join(R, "orbits_transformer_pooled.csv"), index=False)
    print(pooled.to_string())
    v = {c: dict(median_ratio=float(pooled[pooled.criterion == c].ratio.median()), H_F5a_pass=bool(pooled[pooled.criterion == c].ratio.median() <= 0.5)) for c in ("zero", "mean")}
    json.dump(v, open(os.path.join(R, "F7_verdicts.json"), "w"), indent=1); print(v)


if __name__ == "__main__":
    main()
