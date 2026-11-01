import json
rows = []
def add(task, arch, regime, seed, steps, lr=1e-3, wd=1.0, pen=None, extra=None, **model):
    hp = dict(steps=steps, lr=lr, wd=wd, evals=50, patience=5)
    if model: hp['model'] = model
    if pen: hp['pen'] = pen
    if extra: hp.update(extra)
    rows.append(dict(id=f'p2_{task}_{arch}_{regime}_{seed}_{steps}_{lr}_{wd}' + ''.join(f'_{k}{v}' for k, v in model.items()) + (f"_{pen['kind']}{pen['lam']}" if pen else ''),
                     task=task, arch=arch, regime=regime, seed=seed, hp=hp))
# clock vs pizza, 8 seeds, both tasks (pilot seeds only)
for seed in range(900, 908):
    for t, st in (('mod59', 15000), ('mod113', 12000)):
        for al in (0.0, 1.0): add(t, 'tf1', 'clean', seed, st, alpha=al)
# non-abelian / hard cases
for lr, wd in ((1e-3, 1.0), (3e-3, 1.0), (1e-3, 3.0)): add('S5', 'tf1', 'clean', 900, 30000, lr=lr, wd=wd)
add('S5', 'mlp2', 'clean', 900, 30000, lr=3e-3, wd=1.0); add('S5', 'tf2', 'clean', 900, 30000, lr=3e-3, wd=1.0)
add('D12', 'tf1', 'clean', 900, 20000); add('D12', 'tf2', 'clean', 901, 20000, lr=3e-3)
add('comp60', 'tf1', 'clean', 900, 25000); add('permcomp5', 'mlp2', 'clean', 901, 8000); add('permcomp5', 'tf2', 'clean', 900, 12000, lr=3e-3)
add('dyck2', 'tf2', 'clean', 900, 20000, lr=3e-4); add('induction', 'tf2', 'clean', 900, 8000); add('induction', 'lstm', 'clean', 900, 8000); add('induction', 'mlp2', 'clean', 900, 8000)
# regimes on mod59 (fast)
for seed in (900, 901):
    for reg, wd in (('scrambled', 1.0), ('hybrid', 1.0), ('nograk', 0.0), ('contaminated', 0.0)): add('mod59', 'tf1', reg, seed, 15000, wd=wd)
    add('mod59', 'tf1', 'destroy', seed, 15000, pen=dict(kind='destroy', lam=1.0, c=1.0)); add('mod59', 'tf1', 'destroy', seed, 15000, pen=dict(kind='destroy', lam=0.1, c=1.0))
    add('mod59', 'tf1', 'induce', seed, 15000, wd=0.0, pen=dict(kind='induce', lam=1.0)); add('mod59', 'tf1', 'induce', seed, 15000, wd=0.0, pen=dict(kind='induce', lam=0.1))
    add('mod59', 'tf1', 'transplant', seed, 15000, wd=0.0)
for a in ('tf8', 'tf12'): add('mod59', a, 'clean', 900, 15000, lr=5e-4)
add('mod59', 'lstm', 'clean', 900, 10000); add('permcomp5', 'lstm', 'clean', 900, 8000)
with open('bench/pilot_grid2.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(len(rows))
