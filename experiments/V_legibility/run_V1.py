"""V1 — train the legibility ensemble (SPEC §V1) with checkpoints every 5 % of the step budget.
Resumable (skips models whose final checkpoint exists). Writes results/V1_models.csv, ckpt/<model>/step_*.pt."""
import os, sys, csv, json, time, itertools, argparse, shutil, subprocess, random
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


# observed peak allocator use on the 5070 (8 GB): mlp ≤ 0.4 GB, tf2 ≈ 1.2 GB, tf3 ≈ 1.8 GB, tf4 ≈ 4.3 GB; OOM retry covers misses
NEED_MB = {"mlp": 700, "tf2": 3500, "tf3": 3500, "tf4": 4000}
TF_LOCK = os.path.join(CK, ".tf_gpu.lock")   # at most ONE transformer trains at a time (two tf2 sort models OOM'd each
                                             # other 13× in lock-step on the shared 8 GB card, 2026-09-08 22:14)


def tf_lock_try():
    """Non-blocking: True if this process now holds the transformer GPU lock (a dead holder's lock is reclaimed)."""
    for _ in range(2):
        try:
            os.makedirs(TF_LOCK, exist_ok=False); open(os.path.join(TF_LOCK, "pid"), "w").write(str(os.getpid())); return True
        except FileExistsError:
            try: pid = int(open(os.path.join(TF_LOCK, "pid")).read())
            except (ValueError, FileNotFoundError): pid = None
            if pid is not None and _alive(pid) and pid != os.getpid(): return False
            if pid == os.getpid(): return True
            shutil.rmtree(TF_LOCK, ignore_errors=True)
    return False


def tf_lock_release():
    try:
        if int(open(os.path.join(TF_LOCK, "pid")).read()) == os.getpid(): shutil.rmtree(TF_LOCK, ignore_errors=True)
    except (ValueError, FileNotFoundError): pass


def gpu_free_mb():
    try: return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().split()[0])
    except Exception: return 10 ** 9


def _alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def acquire(name):
    """Per-model lock (atomic mkdir) holding the owner's PID; a lock whose owner is dead (or a PID-less lock older than
    2 min from the pre-PID code) is stale and taken over."""
    d = os.path.join(CK, name); os.makedirs(d, exist_ok=True); lock = os.path.join(d, ".lock")
    for _ in range(2):
        try:
            os.makedirs(lock, exist_ok=False); open(os.path.join(lock, "pid"), "w").write(str(os.getpid())); return True
        except FileExistsError:
            pf = os.path.join(lock, "pid")
            if os.path.exists(pf):
                try: pid = int(open(pf).read())
                except ValueError: return False
                if _alive(pid): return False
            elif time.time() - os.path.getmtime(lock) < 120: return False
            shutil.rmtree(lock, ignore_errors=True)
    return False


def main():
    # --shard i n: this process trains plan()[i::n] (disjoint shards run as parallel units; AMENDMENTS V#1 2026-09-08)
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, nargs=2, default=(0, 1)); a = ap.parse_args()
    rows = plan(); out = os.path.join(R, "V1_models.csv")
    done = {r["name"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    mine = rows[a.shard[0]::a.shard[1]]
    print(len(rows), "models planned,", len(done), "done;", len(mine), "in shard", a.shard, flush=True)
    # own shard first, then steal any model no process has locked (atomic mkdir lock per model; a shard that starts
    # late or dies does not stall the ensemble)
    def read_done():
        return {r["name"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    # Multi-pass: a transformer whose GPU lock is held by another shard is skipped (its model lock released) and the
    # shard takes the next MLP instead; passes repeat until nothing is left. `done` is re-read from the CSV before
    # every model so a finished model is never retrained or double-logged (V AMENDMENTS #2 addendum 22:30).
    while True:
        progressed = False; deferred = False
        for row in mine + [r for r in rows if r not in mine]:
            if row["name"] in read_done(): continue
            big = row["arch"].startswith("tf")
            if big and not tf_lock_try():
                deferred = True; continue
            if not acquire(row["name"]):
                if big: tf_lock_release()
                continue
            try:
                hp = os.path.join(CK, row["name"], "history.json")
                if os.path.exists(hp) and not row["untrained"]:      # trained before a crash, CSV row lost: rebuild the row from history
                    hist = json.load(open(hp)); res = dict(steps=STEPS, eval_acc=hist[-1]["eval_acc"], train_acc=hist[-1]["train_acc"], wall=float("nan"), n_ckpt=len(hist))
                else:
                    res = None
                    for attempt in range(40):                          # shared GPU: wait for headroom, retry on OOM (V AMENDMENTS #1)
                        torch.cuda.empty_cache()                       # release our own cache first (three shards deadlocked on each
                        need = NEED_MB.get(row["arch"][:3], 2500)      # other's cached memory at 0 % util, V AMENDMENTS #2)
                        while gpu_free_mb() < need:                    # headroom sized per architecture
                            torch.cuda.empty_cache(); time.sleep(60)
                        try: res = train_one(row); torch.cuda.empty_cache(); break
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache(); print(f"  OOM on {row['name']} (attempt {attempt}), retrying", flush=True); time.sleep(60 + random.random() * 120)
                    if res is None: shutil.rmtree(os.path.join(CK, row["name"], ".lock"), ignore_errors=True); continue
            finally:
                if big: tf_lock_release()
            if row["name"] in read_done(): continue                    # another shard logged it meanwhile (crash-recovery race)
            row.update(res)
            new = not os.path.exists(out)
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new: w.writeheader()
                w.writerow(row)
            print(row, flush=True); progressed = True
        if not deferred: break
        if not progressed: time.sleep(60 + random.random() * 60)
    print("done", flush=True)


if __name__ == "__main__":
    main()
