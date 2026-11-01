"""Pilot grid (seeds 900-901 only; never confirmatory).  Calibrates (steps, lr, wd) per (task, arch)."""
import json, sys
rows = []
def add(task, arch, regime, seed, steps, lr=1e-3, wd=1.0, **model):
    hp = dict(steps=steps, lr=lr, wd=wd, evals=50, patience=5)
    if model: hp['model'] = model
    rows.append(dict(id=f'pilot_{task}_{arch}_{regime}_{seed}_{steps}_{lr}_{wd}' + ''.join(f'_{k}{v}' for k, v in model.items()), task=task, arch=arch, regime=regime, seed=seed, hp=hp))
for seed in (900,):
    for task in ('mod59', 'mod97', 'comp60', 'comp96'): add(task, 'tf1', 'clean', seed, 8000)
    for task in ('A5', 'D12', 'S5'): add(task, 'tf2', 'clean', seed, 12000); add(task, 'mlp2', 'clean', seed, 12000)
    for task in ('permcomp5', 'sorting', 'copying', 'dyck2', 'parity', 'symfree'):
        for arch in ('tf2', 'lstm', 'mlp2'): add(task, arch, 'clean', seed, 8000)
    for arch in ('mlp2', 'tf2', 'tf4', 'lstm', 'mamba'): add('mod59', arch, 'clean', seed, 10000)
    for al in (0.0, 1.0): add('mod113', 'tf1', 'clean', seed, 12000, alpha=al); add('mod59', 'tf1', 'clean', seed, 12000, alpha=al)
with open('bench/pilot_grid.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(len(rows))
