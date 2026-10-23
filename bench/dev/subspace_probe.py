import sys, numpy as np, torch
sys.path.insert(0, ".")
from bench.tasks import AdditionTask
from whitebox_ee.llm import HFSystem
from whitebox_ee.systems import Patch
from whitebox_ee.probe import member_from_labels
task = AdditionTask(n_digits=2, n_shots=3)
rng = np.random.default_rng(0)
X = task.sample(300, rng, carries=(0, 1)) + task.sample(300, rng, carries=(1, 1))
sysm = HFSystem("Qwen/Qwen2.5-1.5B", task=task, layers=[22, 24, 26, 27], roles=["pre_ans_d0"], batch_size=64)
sysm.behaviours = ["out0", "out1"]
lg, st = sysm.run(X)
L = task.labels(X)
src = rng.permutation(len(X))
d = sysm.d_model
ids = task.class_token_ids(sysm.tok)
Wu = sysm.model.get_output_embeddings().weight[ids].detach().float().cpu().numpy()
def follow(U, s):
    lg2, _ = sysm.run(X, [Patch(s, U.astype(np.float32), st[s][src])], sites=[])
    return float((lg2["out0"].argmax(-1) == lg["out0"][src].argmax(-1)).mean()), float((lg2["out0"].argmax(-1) != lg["out0"].argmax(-1)).mean())
for s in sysm.sites:
    H = st[s] - st[s].mean(0)
    _, S, Vt = np.linalg.svd(H, full_matrices=False)
    out = []
    for m in (8, 32, 128, 512):
        out.append((f"pc{m}", follow(Vt[:m].T, s)))
    Uu = np.linalg.qr(Wu.T)[0]
    out.append(("unembed10", follow(Uu, s)))
    for v in ("r0", "s0"):
        mem = member_from_labels(H, L[v], s)
        out.append((f"ridge_{v}_k{mem.U.shape[1]}", follow(mem.U, s)))
        mem = member_from_labels(H, L[v], s, n_pca=16)
        out.append((f"ridge_{v}_pca16_k{mem.U.shape[1]}", follow(mem.U, s)))
    print(s, [(n, round(f, 2), round(c, 2)) for n, (f, c) in out], flush=True)
