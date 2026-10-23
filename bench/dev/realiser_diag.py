"""Development diagnostic: delta vs plain probe realisers on a small LLM (not confirmatory)."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from bench.tasks import AdditionTask
from whitebox_ee.llm import HFSystem
from whitebox_ee.probe import Probe, Node
from whitebox_ee.propose import propose_probes
from whitebox_ee.filter import Thresholds, quotient_test
from whitebox_ee.info import entropy
task = AdditionTask(n_digits=2, n_shots=3)
rng = np.random.default_rng(0)
X = task.sample(300, rng, carries=(0, 1)) + task.sample(300, rng, carries=(1, 1))
layers = [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
roles = ["pre_ans_d1", "pre_ans_d0"]
sysm = HFSystem("Qwen/Qwen2.5-1.5B", task=task, layers=layers, roles=[r for r in task.roles if r in roles], batch_size=64)
sysm.behaviours = ["out0", "out1"]
probe = Probe(sysm, X)
L = task.labels(X)
labels = {k: L[k] for k in ["s0", "c0", "s1", "r0", "r1"]}
th = Thresholds(n_random_realisers=8, n_pairs=256)
H_X = sum(entropy(v) for v in probe.inputs.values())
for mode in ("plain", "delta"):
    mem = propose_probes(probe, labels, mode=mode, sites=[s for s in sysm.sites if not s.startswith("L0@")])
    print(f"== {mode}: {len(mem)} members", flush=True)
    for m in mem:
        nd = Node("x", "concept", [m], m.values(probe.states[m.site]), probe.site_layer(m.site))
        q = quotient_test(probe, nd, H_X, th, {}, rng)
        print(f"{m.source:14s} {m.site:16s} k={m.U.shape[1]:2d} A={q.A:.3f} eta={q.eta:.3f} dQ={q.delta_Q:.3f} pass={q.passed} {q.reasons[:1]}", flush=True)
