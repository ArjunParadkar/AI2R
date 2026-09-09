"""V1 (large arms, HPCC) — extends the legibility ensemble with the arms the laptop could not train: MLP widths 1024 / 2048,
transformers with 6 / 8 layers (d = 128), 3 seeds each, on the same tasks / regularisation grid / checkpoint schedule as
``experiments/V_legibility/run_V1.py`` (its ``train_one`` is reused unchanged, so checkpoints and CSV rows are format-identical
and V2/V3/V5/V6 run on them without changes). Sharded: ``--shard i n`` trains plan()[i::n]; every shard appends to the same
CSV under the per-model mkdir lock of run_V1 (safe on a shared filesystem).
Usage:  python v1_ensemble_large.py --out results/v1_large --shard 0 8"""
import os, sys, csv, itertools, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
VL = os.path.join(ROOT, "experiments", "V_legibility"); sys.path.insert(0, VL)
import run_V1
ARCHS_LARGE = [("mlp1024", dict(arch="mlp", width=1024)), ("mlp2048", dict(arch="mlp", width=2048)), ("tf6", dict(arch="tf", n_layers=6)), ("tf8", dict(arch="tf", n_layers=8))]


def plan_large():
    rows = []
    for ti, task in enumerate(run_V1.TASKS):
        for ai, (l1, wd, dr) in enumerate(run_V1.GRID):
            aname, spec = ARCHS_LARGE[(ai + ti) % len(ARCHS_LARGE)]
            for seed in (0, 1, 2):
                rows.append(dict(name=f"{task}_{aname}_l1{l1}_wd{wd}_dr{dr}_s{seed}", task=task, arch=aname, l1=l1, wd=wd, dropout=dr, clus=0.0, random_labels=0, untrained=0, seed=seed))
    for task, aname in (("modadd", "mlp1024"), ("sort", "tf6"), ("copy", "mlp2048"), ("modadd", "tf8")):
        for lam in (5.0, 40.0): rows.append(dict(name=f"{task}_{aname}_clus{lam}_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=lam, random_labels=0, untrained=0, seed=0))
        rows.append(dict(name=f"{task}_{aname}_randlab_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=0.0, random_labels=1, untrained=0, seed=0))
        rows.append(dict(name=f"{task}_{aname}_untrained_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=0.0, random_labels=0, untrained=1, seed=0))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/v1_large"); ap.add_argument("--shard", type=int, nargs=2, default=(0, 1)); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); run_V1.R = a.out; run_V1.CK = os.path.join(a.out, "ckpt"); os.makedirs(run_V1.CK, exist_ok=True)
    run_V1.ARCHS = run_V1.ARCHS + ARCHS_LARGE                    # build() looks the spec up by name through ARCHS
    rows = plan_large(); mine = rows[a.shard[0]::a.shard[1]]; csv_p = os.path.join(a.out, "V1_models.csv")
    done = {r["name"] for r in csv.DictReader(open(csv_p))} if os.path.exists(csv_p) else set()
    print(len(rows), "planned", len(done), "done", len(mine), "in shard", a.shard, flush=True)
    for row in mine:
        if row["name"] in done or not run_V1.acquire(row["name"]): continue
        res = run_V1.train_one(row, device="cuda"); print(res, flush=True)
        new = not os.path.exists(csv_p)
        with open(csv_p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res)); (w.writeheader() if new else None); w.writerow(res)


if __name__ == "__main__":
    main()
