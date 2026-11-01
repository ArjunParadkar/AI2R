"""E4 driver: train the ensemble, measure legibility profile + run the fixed discovery pipeline.
Resumable: skips model ids already in results/e4_models.csv. `--smoke` = 4 cells."""
import argparse, json, os, sys, time, csv
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e4lib.tasks import Task
from e4lib.models import build_model
from e4lib.train import train, ARMS
from e4lib.cluster_loss import block_mask_
from e4lib.legibility import legibility_profile

ARCHS = ["mlp", "tf2", "tf4"]; TASKS = ["modadd", "sort", "copy"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def grid():
    cells = []
    for a in ARCHS:
        for t in TASKS:
            for arm in ARMS: cells.append(dict(arch=a, task=t, arm=arm, kind="trained"))
    for a in ARCHS: cells.append(dict(arch=a, task="modadd", arm="none", kind="randlabel"))
    for a in ARCHS:
        for t in TASKS: cells.append(dict(arch=a, task=t, arm="none", kind="untrained"))
    for a in ARCHS: cells.append(dict(arch=a, task="modadd", arm="none", kind="untrained_block"))
    for c in cells: c["id"] = f"{c['kind']}|{c['arch']}|{c['task']}|{c['arm']}"
    return cells

def run_cell(c, device, log):
    task = Task(c["task"], split_seed=0, random_labels=(c["kind"] == "randlabel"))
    arm = dict(ARMS[c["arm"]]); dropout = arm.pop("dropout", 0.0)
    model = build_model(c["arch"], task, dropout=dropout, seed=0)
    info = dict(steps=0, eval_acc=np.nan, train_acc=np.nan, train_time=0.0)
    if c["kind"] in ("trained", "randlabel"):
        tgt = 0.95 if c["task"] == "modadd" else 0.99
        if c["kind"] == "randlabel": tgt = 0.99
        info = train(model, task, steps=4000, device=device, target_acc=tgt, log=log, **arm)
    elif c["kind"] == "untrained_block":
        block_mask_(model)
    model.to(device).eval()
    with torch.no_grad():
        Xev, Yev = task.all("eval"); acc = (model(Xev.to(device)).argmax(-1).cpu() == Yev).float().mean().item()
    learned = acc >= 0.5   # SPEC amendment 2
    tk, tg = task.batch(256, 1000); tk, tg = tk.to(device), tg.to(device)
    disc = [tuple(x.to(device) for x in task.batch(256, 2000 + s)) for s in range(3)]
    t0 = time.time()
    prof = legibility_profile(model, (tk, tg), discovery_batches=disc, null_n=20, seed=0)
    row = dict(c); row.update(info); row.update(acc=acc, chance=task.chance, learned=learned, profile_time=time.time() - t0)
    for k, v in prof.items():
        if k in ("sweep", "runs"): continue
        row[k] = v
    row["S"] = row["S_raw"] if learned else 0.0
    row["jaccard_guarded"] = row["jaccard"] if learned else np.nan
    row["binary_success"] = bool(learned and row["S"] >= 0.75 and row["jaccard"] >= 0.5)
    sweep = [dict(id=c["id"], **r) for r in prof["sweep"]]
    circuits = [dict(id=c["id"], seed=s, keep=json.dumps(np.flatnonzero(r["keep"].numpy()).tolist()), F=r["F"], cost=r["cost"])
                for s, r in enumerate(prof["runs"])]
    return row, sweep, circuits

def append(path, rows):
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--suffix", default=""); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = a.device if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    cells = grid()
    if a.smoke:
        ids = ["trained|mlp|sort|none", "trained|tf2|modadd|none", "trained|tf4|copy|clus40", "untrained_block|tf2|modadd|none"]
        cells = [c for c in cells if c["id"] in ids]
    mp, sp, cp = [os.path.join(OUT, f"e4_{n}{a.suffix}.csv") for n in ("models", "sweep", "circuits")]
    done = set(pd.read_csv(mp)["id"]) if os.path.exists(mp) else set()
    logf = open(os.path.join(OUT, f"run{a.suffix}.log"), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"== E4 run {time.strftime('%F %T')} device={device} cells={len(cells)} done={len(done)}")
    for i, c in enumerate(cells):
        if c["id"] in done: continue
        log(f"[{i+1}/{len(cells)}] {c['id']}"); t0 = time.time()
        row, sweep, circ = run_cell(c, device, log)
        append(mp, [row]); append(sp, sweep); append(cp, circ)
        log(f"   acc {row['acc']:.3f} learned {row['learned']} Q10 {row['Q10']:.3f} (null {row['Q10_null']:.3f}) tw10 {row['tw10']} "
            f"sp {row['sparsity']:.2f} S {row['S']:.3f} F {row['F']:.3f} cost {row['cost']:.0f} J {row['jaccard']:.2f} [{time.time()-t0:.0f}s]")
    log("done")
