"""Pilot / smoke runner: python bench/pilot.py task arch regime seed [steps lr wd]  (pilot seeds 900+ are never confirmatory)"""
import sys, json, time
sys.path.insert(0, '.')
from bench.zoo import *
t, a, r, s = sys.argv[1:5]; steps = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
lr = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3; wd = float(sys.argv[7]) if len(sys.argv) > 7 else (0.0 if r in ('nograk','contaminated') else 1.0)
hp = dict(steps=steps, lr=lr, wd=wd, evals=50, patience=5)
t0 = time.time(); rows = run_config(dict(task=t, arch=a, regime=r, seed=int(s), hp=hp), progress=print)
for row in rows:
    row = dict(row); tr = row.pop('traj', None)
    keys = [k for k in row if k not in ('hp', 'slot_share')]
    print({k: (round(row[k], 4) if isinstance(row[k], float) else row[k]) for k in keys})
print('time', time.time() - t0)
