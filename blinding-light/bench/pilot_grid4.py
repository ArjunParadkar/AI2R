import json
rows = []
def add(task, arch, regime, seed, steps, lr=1e-3, wd=1.0, pen=None, **model):
    hp = dict(steps=steps, lr=lr, wd=wd, evals=50, patience=5)
    if model: hp['model'] = model
    if pen: hp['pen'] = pen
    rows.append(dict(id=f'p4_{task}_{arch}_{regime}_{seed}_{steps}_{lr}_{wd}' + (f"_{pen['kind']}{pen['lam']}" if pen else ''), task=task, arch=arch, regime=regime, seed=seed, hp=hp))
for seed in (950, 951):
    for lam in (0.1, 1.0):
        add('mod59', 'tf1', 'destroy', seed, 15000, pen=dict(kind='destroy', lam=lam, c=1.0)); add('mod59', 'tf1', 'induce', seed, 15000, wd=0.0, pen=dict(kind='induce', lam=lam))
    add('mod59', 'tf1', 'hybrid', seed, 40000); add('mod59', 'tf1', 'hybrid', seed, 40000, lr=3e-3)
    add('mod59', 'tf1', 'nograk', seed, 15000, wd=0.0); add('mod59', 'tf1', 'transplant', seed, 15000, wd=0.0)
add('mod59', 'lstm', 'clean', 950, 10000); add('permcomp5', 'lstm', 'clean', 950, 8000); add('induction', 'lstm', 'clean', 950, 8000)
add('induction', 'mlp2', 'clean', 950, 30000, lr=3e-3); add('induction', 'tf2', 'clean', 950, 30000, lr=3e-3); add('A5', 'mlp2', 'clean', 950, 20000, lr=3e-3)
add('mod113', 'tf1', 'destroy', 950, 12000, pen=dict(kind='destroy', lam=1.0, c=1.0)); add('mod113', 'tf1', 'induce', 950, 12000, wd=0.0, pen=dict(kind='induce', lam=1.0)); add('mod113', 'tf1', 'transplant', 950, 12000, wd=0.0)
add('permcomp5', 'tf2', 'clean', 951, 12000, lr=3e-3, wd=1.0); add('permcomp5', 'mlp2', 'scrambled', 950, 8000); add('S5', 'mlp2', 'scrambled', 950, 15000, lr=3e-3)
with open('bench/pilot_grid4.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(len(rows))
