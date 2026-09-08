"""V1 — train the legibility ensemble (SPEC §V1) with checkpoints every 5 % of the step budget.
Resumable (skips models whose final checkpoint exists). Writes results/V1_models.csv, ckpt/<model>/step_*.pt."""
import os, sys, csv, json, time, itertools, argparse
import numpy as np, torch, torch.nn.functional as F
from whiteboxtopo.legibility.tasks import Task, accuracy
from whiteboxtopo.legibility.e4models import MLP, TinyTF
from whiteboxtopo.legibility.cluster_loss import clusterability_loss, block_mask_
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results"); CK = os.path.join(HERE, "ckpt")
os.makedirs(R, exist_ok=True); os.makedirs(CK, exist_ok=True)
TASKS = ("modadd", "sort", "copy")
ARCHS = [("mlp64", dict(arch="mlp", width=64)), ("mlp128", dict(arch="mlp", width=128)), ("mlp256", dict(arch="mlp", width=256)),
         ("mlp512", dict(arch="mlp", width=512)), ("tf2", dict(arch="tf", n_layers=2)), ("tf3", dict(arch="tf", n_layers=3)), ("tf4", dict(arch="tf", n_layers=4))]
GRID = list(itertools.product((0.0, 1e-4, 1e-3), (0.0, 0.1, 1.0), (0.0, 0.2)))   # 18 arms: (l1, wd, dropout)
STEPS, N_CK = 20000, 20


def build(spec, task, dropout, seed):
    if spec["arch"] == "mlp": return MLP(task, width=spec["width"], dropout=dropout, seed=seed)
    return TinyTF(task, n_layers=spec["n_layers"], d=128, heads=4, mlp=256, dropout=dropout, seed=seed)


def plan():
    """Fractional design: arm i on task t uses architectures ARCHS[(i + t) % 7] and ARCHS[(i + t + 3) % 7]."""
    rows = []
    for ti, task in enumerate(TASKS):
        for ai, (l1, wd, dr) in enumerate(GRID):
            for k in (0, 3):
                aname, spec = ARCHS[(ai + ti + k) % len(ARCHS)]
                rows.append(dict(name=f"{task}_{aname}_l1{l1}_wd{wd}_dr{dr}_s0", task=task, arch=aname, l1=l1, wd=wd, dropout=dr, clus=0.0, random_labels=0, untrained=0, seed=0))
    # clusterability-loss models (Golechha recipe): 5 (task, arch) pairs x lambda in {5, 40}
    for j, (task, aname) in enumerate([("modadd", "mlp128"), ("sort", "tf2"), ("copy", "mlp256"), ("modadd", "tf3"), ("sort", "mlp64")]):
        for lam in (5.0, 40.0):
            rows.append(dict(name=f"{task}_{aname}_clus{lam}_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=lam, random_labels=0, untrained=0, seed=0))
    for j, (task, aname) in enumerate([("modadd", "mlp128"), ("sort", "tf2"), ("copy", "mlp64"), ("modadd", "tf4"), ("sort", "mlp512"), ("copy", "tf3")]):
        rows.append(dict(name=f"{task}_{aname}_randlab_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=0.0, random_labels=1, untrained=0, seed=0))
        rows.append(dict(name=f"{task}_{aname}_untrained_s0", task=task, arch=aname, l1=0.0, wd=0.0, dropout=0.0, clus=0.0, random_labels=0, untrained=1, seed=0))
    return rows


def train_one(row, device="cuda", log=print):
    task = Task(row["task"], random_labels=bool(row["random_labels"]))
    spec = dict(ARCHS)[row["arch"]]; model = build(spec, task, row["dropout"], row["seed"]).to(device)
    d = os.path.join(CK, row["name"]); os.makedirs(d, exist_ok=True)
    Xtr, Ytr = task.all("train"); Xev, Yev = task.all("eval"); Xtr, Ytr, Xev, Yev = Xtr.to(device), Ytr.to(device), Xev.to(device), Yev.to(device)
    every = STEPS // N_CK; hist = []
    if row["untrained"]:
        torch.save(model.state_dict(), os.path.join(d, f"step_{STEPS:06d}.pt"))
        model.eval()
        with torch.no_grad(): acc = accuracy(model(Xev), Yev)
        return dict(steps=0, eval_acc=acc, train_acc=acc, wall=0.0, n_ckpt=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=row["wd"]); g = torch.Generator().manual_seed(row["seed"]); t0 = time.time()
    step = 0
    for step in range(1, STEPS + 1):
        model.train(); idx = torch.randint(0, len(Xtr), (min(256, len(Xtr)),), generator=g)
        logits = model(Xtr[idx]); loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Ytr[idx].reshape(-1))
        if row["l1"]: loss = loss + row["l1"] * sum(p.abs().sum() for p in model.parameters() if p.dim() >= 2)
        if row["clus"]: loss = loss + row["clus"] * clusterability_loss(model.clusterable)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % every == 0:
            model.eval()
            with torch.no_grad(): acc = accuracy(model(Xev), Yev); tr_acc = accuracy(model(Xtr[:2048]), Ytr[:2048])
            torch.save(model.state_dict(), os.path.join(d, f"step_{step:06d}.pt")); hist.append(dict(step=step, loss=loss.item(), eval_acc=acc, train_acc=tr_acc))
            log(f"  {row['name']} step {step} loss {loss.item():.3f} eval {acc:.3f}")
    json.dump(hist, open(os.path.join(d, "history.json"), "w"))
    return dict(steps=step, eval_acc=hist[-1]["eval_acc"], train_acc=hist[-1]["train_acc"], wall=time.time() - t0, n_ckpt=len(hist))


def main():
    # --shard i n: this process trains plan()[i::n] (disjoint shards run as parallel units; AMENDMENTS V#1 2026-09-08)
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, nargs=2, default=(0, 1)); a = ap.parse_args()
    rows = plan(); out = os.path.join(R, "V1_models.csv")
    done = {r["name"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    mine = rows[a.shard[0]::a.shard[1]]
    print(len(rows), "models planned,", len(done), "done;", len(mine), "in shard", a.shard, flush=True)
    # own shard first, then steal any model no process has locked (atomic mkdir lock per model; a shard that starts
    # late or dies does not stall the ensemble)
    for row in mine + [r for r in rows if r not in mine]:
        if row["name"] in done: continue
        lock = os.path.join(CK, row["name"], ".lock")
        try: os.makedirs(lock, exist_ok=False)
        except FileExistsError: continue
        hp = os.path.join(CK, row["name"], "history.json")
        if os.path.exists(hp) and not row["untrained"]:      # trained before a crash, CSV row lost: rebuild the row from history
            hist = json.load(open(hp)); res = dict(steps=STEPS, eval_acc=hist[-1]["eval_acc"], train_acc=hist[-1]["train_acc"], wall=float("nan"), n_ckpt=len(hist))
        else:
            res = train_one(row)
        row.update(res)
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
        print(row, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
