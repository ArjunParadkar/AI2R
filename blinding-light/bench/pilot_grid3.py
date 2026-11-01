import json
rows = []
def add(task, seed, steps, al, arch='tf1'):
    rows.append(dict(id=f'p3_{task}_{arch}_a{al}_{seed}', task=task, arch=arch, regime='clean', seed=seed, hp=dict(steps=steps, lr=1e-3, wd=1.0, evals=50, patience=5, model=dict(alpha=al))))
for seed in range(910, 926):
    for al in (0.0, 1.0): add('mod59', seed, 15000, al)
for seed in range(910, 918):
    for al in (0.0, 1.0): add('mod113', seed, 12000, al)
with open('bench/pilot_grid3.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(len(rows))
