"""Development run (NOT confirmatory): full pipeline on a small LLM on addition.
Models used here are excluded from every pre-registered model list."""
import argparse, json, sys, time
import numpy as np
import torch
sys.path.insert(0, ".")
from bench.tasks import AdditionTask
from whitebox_ee.llm import HFSystem
from whitebox_ee.probe import Probe
from whitebox_ee.propose import propose_probes
from whitebox_ee.discover import discover
from whitebox_ee.graph import label_nodes
from whitebox_ee.generality import compute_activity, generality
from whitebox_ee.filter import Thresholds

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--digits", type=int, default=2)
ap.add_argument("--layer_step", type=int, default=2)
ap.add_argument("--out", default="runs/dev/llm_dev.json")
a = ap.parse_args()
task = AdditionTask(n_digits=a.digits, n_shots=3)
rng = np.random.default_rng(0)
nd = a.digits
pats = [tuple(list(p) + [1]) for p in np.ndindex(*([2] * (nd - 1)))]
X = []
for p in pats:
    X += task.sample(a.n // len(pats), rng, carries=p)
tmp = HFSystem(a.model, task=task, layers=[0], dtype="bfloat16")
L = tmp.n_layers
cut = int(np.ceil(2 * L / 3))
layers = sorted(set(list(range(0, cut, 4)) + list(range(cut, L + 1))))
del tmp; torch.cuda.empty_cache()
sysm = HFSystem(a.model, task=task, layers=layers, dtype="bfloat16", batch_size=64)
sysm.behaviours = [f"out{j}" for j in range(nd)]          # carry-out constant (=1): dropped
t0 = time.time()
probe = Probe(sysm, X)
tg = task.targets(X)
acc = {b: float((probe.read[b] == tg[b]).mean()) for b in sysm.behaviours}
full = float(np.mean(np.all([probe.read[b] == tg[b] for b in sysm.behaviours], 0)))
print("acc", acc, "full", full, "probe %.0fs" % (time.time() - t0), flush=True)
labels = {k: v for k, v in task.labels(X).items() if not k.startswith(f"c{nd-1}")}
mem = propose_probes(probe, labels, verbose=True)
print(len(mem), "candidates", flush=True)
th = Thresholds(n_random_realisers=10)
G, probe, rec = discover(sysm, X, mem, th=th, probe=probe, verbose=True, stability=False, minimal_sets=False)
lab = label_nodes(G, labels, len(X))
A = compute_activity(probe, G, th, rng, rows=np.arange(min(300, len(X))))
g = generality(G, A)
res = {"acc": acc, "full": full, "n_cand": len(mem), "runtime_s": time.time() - t0, "rows": probe.n_runs,
       "nodes": {c: {"label": lab[c], "site": G.nodes[c].primary.site, "parents": [lab.get(p) or p for p in G.parents[c]],
                     "g": g[c], "iia": G.certificates[c].get("iia", {}).get("iia")} for c in G.concepts()},
       "outputs": {o: [lab.get(p) or p for p in G.parents[o]] for o in G.outputs()},
       "dropped": rec["dropped"], "verdicts": {}}
from collections import Counter
res["verdicts"] = dict(Counter(v["verdict"][:24] for v in rec["candidates"].values()))
for c, v in sorted(res["nodes"].items(), key=lambda kv: -kv[1]["g"]):
    print(c, v)
print(res["outputs"]); print(res["verdicts"]); print("runtime %.0fs rows %d" % (res["runtime_s"], res["rows"]))
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=str)
