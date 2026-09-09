"""V2 (structure) + V3 (discovery cost) on the V1 final checkpoints; V5 uses the same on every
checkpoint (--all-ckpts, structure only + discovery at 5 checkpoints). Writes results/V2_structure.csv,
V3_discovery.csv, V5_drift.csv. Resumable."""
import os, sys, csv, glob, json, time, argparse
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mejiro_dober import plan, build, ARCHS, CK, R, STEPS, N_CK
from whiteboxtopo.legibility.tasks import Task, accuracy
from whiteboxtopo import graphs, discovery
from whiteboxtopo.legibility.models import node_importance
DENS = (0.01, 0.05, 0.1)


def load(row, step, device):
    task = Task(row["task"], random_labels=bool(row["random_labels"])); spec = dict(ARCHS)[row["arch"]]
    m = build(spec, task, 0.0, row["seed"]).to(device)
    m.load_state_dict(torch.load(os.path.join(CK, row["name"], f"step_{step:06d}.pt"), map_location=device)); m.eval()
    return m, task


def structure(m, task, seed=0, device="cuda", null=True):
    x, y = task.batch(512, seed, "eval"); x, y = x.to(device), y.to(device)
    E = graphs.attribution_graph(m, (x, y), raw=True); A = graphs.symmetrise(E); N = m.n_nodes
    out = dict(N=N, sparsity_nodes=graphs.sparsity(node_importance(m, x)), edge_sparsity=graphs.sparsity(A))
    with torch.no_grad(): out["accuracy"] = accuracy(m(x), y)
    for dn in DENS:
        B = graphs.threshold_density(A, dn); Q, memb = graphs.modularity(B); out[f"Q_{dn}"] = Q; out[f"tw_{dn}"] = graphs.treewidth_ub(B)
        out[f"n_comm_{dn}"] = int(len(set(memb)))
        if null:
            q0, sd = graphs.rewired_null(B, n=20, seed=seed); out[f"Qnull_{dn}"] = q0; out[f"Qnull_sd_{dn}"] = sd; out[f"Qexcess_{dn}"] = Q - q0
    return out


def disc(m, task, device="cuda"):
    runs = []
    for s in range(3):
        x, y = task.batch(256, 100 + s, "eval"); r = discovery.find_circuit(m, (x.to(device), y.to(device)), faithfulness=0.9, seed=s); runs.append(r)
    js = [discovery.jaccard(runs[i]["circuit"], runs[j]["circuit"]) for i in range(3) for j in range(i + 1, 3)]
    return dict(size=np.mean([r["size"] for r in runs]), frac=np.mean([r["frac"] for r in runs]), F=np.nanmean([r["faithfulness"] for r in runs]),
                cost=np.mean([r["cost"] for r in runs]), wall=np.mean([r["wall"] for r in runs]), success=int(all(r["success"] for r in runs)),
                success_any=int(any(r["success"] for r in runs)), jaccard=float(np.mean(js)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--all-ckpts", action="store_true"); a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(R, "V1_models.csv")))]
    dev = "cuda"
    if not a.all_ckpts:
        p2, p3 = os.path.join(R, "V2_structure.csv"), os.path.join(R, "V3_discovery.csv")
        done = {r["name"] for r in csv.DictReader(open(p3))} if os.path.exists(p3) else set()
        for row in rows:
            if row["name"] in done: continue
            for k in ("random_labels", "untrained", "seed"): row[k] = int(row[k])
            m, task = load(row, STEPS, dev); t0 = time.time()
            s2 = structure(m, task); s2.update(name=row["name"], task=row["task"], arch=row["arch"])
            s3 = disc(m, task); s3.update(name=row["name"], task=row["task"], arch=row["arch"], wall_total=time.time() - t0)
            for p, r in ((p2, s2), (p3, s3)):
                new = not os.path.exists(p)
                with open(p, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(r));
                    if new: w.writeheader()
                    w.writerow(r)
            print(row["name"], {k: round(v, 3) if isinstance(v, float) else v for k, v in s2.items() if k.startswith("Q")}, s3["size"], s3["F"], flush=True)
    else:
        p5 = os.path.join(R, "V5_drift.csv"); done = {(r["name"], int(r["step"])) for r in csv.DictReader(open(p5))} if os.path.exists(p5) else set()
        every = STEPS // N_CK; dsteps = {every * k for k in (1, 5, 10, 15, 20)}
        for row in rows:
            for k in ("random_labels", "untrained", "seed"): row[k] = int(row[k])
            if row["untrained"]: continue
            hist = {h["step"]: h for h in json.load(open(os.path.join(CK, row["name"], "history.json")))}
            for step in sorted(hist):
                if (row["name"], step) in done: continue
                m, task = load(row, step, dev)
                r = structure(m, task, null=True); r.update(name=row["name"], task=row["task"], arch=row["arch"], step=step, frac=step / STEPS,
                                                          loss=hist[step]["loss"], eval_acc=hist[step]["eval_acc"], random_labels=row["random_labels"])
                if step in dsteps: r.update({f"disc_{k}": v for k, v in disc(m, task).items()})
                new = not os.path.exists(p5)
                with open(p5, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(r), extrasaction="ignore")
                    if new: w.writeheader()
                    w.writerow(r)
                print(row["name"], step, round(r["Qexcess_0.1"], 3), flush=True)
    print("done")


if __name__ == "__main__":
    main()
