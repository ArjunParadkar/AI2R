import sys, numpy as np
sys.path.insert(0, ".")
from bench.tasks import AdditionTask
from whitebox_ee.llm import HFSystem
from whitebox_ee.systems import Patch
task = AdditionTask(n_digits=2, n_shots=3)
rng = np.random.default_rng(0)
X = task.sample(64, rng, carries=(0, 1)) + task.sample(64, rng, carries=(1, 1))
sysm = HFSystem("Qwen/Qwen2.5-1.5B", task=task, layers=[0, 6, 10, 14, 18, 22, 26], batch_size=64)
sysm.behaviours = ["out0", "out1"]
lg, st = sysm.run(X)
print("roles", sysm.roles)
src = rng.permutation(len(X))
d = sysm.d_model
for s in sysm.sites:
    P = Patch(s, np.eye(d, dtype=np.float32), st[s][src])
    lg2, _ = sysm.run(X, [P], sites=[])
    ch = {b: float((lg2[b].argmax(-1) != lg[b].argmax(-1)).mean()) for b in lg}
    follow = {b: float((lg2[b].argmax(-1) == lg[b][src].argmax(-1)).mean()) for b in lg}
    print(s, "changed", ch, "follows source", follow, flush=True)
