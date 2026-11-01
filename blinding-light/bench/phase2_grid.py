"""Phase 2 confirmatory grid.  python bench/phase2_grid.py -> bench/phase2_grid.jsonl.  Frozen (sha256 in prereg/PHASE_2.md) before any confirmatory run.
Seeds 1-10 (1-5 for LSTM / Mamba / tf8 / tf12).  Pilot seeds >= 900 are never confirmatory.  Authors: Arjun Paradkar and Reva Paradkar."""
import json, sys
DESTROY = {10: dict(kind='destroy', lam=10.0, c=1.0, ramp=2000), 30: dict(kind='destroy', lam=30.0, c=1.0, ramp=2000)}   # two doses (pilot: 10 weak, 30 variable, 100 collapses training)
INDUCE = dict(kind='induce', lam=0.1, ramp=2000)
rows = []
S10, S5 = range(1, 11), range(1, 6)


def add(block, task, arch, regime, seeds, steps, lr=1e-3, wd=None, pen=None, donor=False, **model):
    wd = (0.0 if regime in ('nograk', 'contaminated', 'induce', 'transplant') else 1.0) if wd is None else wd
    for sd in seeds:
        hp = dict(steps=steps, lr=lr, wd=wd, evals=50, patience=5)
        if model: hp['model'] = model
        if pen: hp['pen'] = pen
        if donor: hp['donor'] = dict(steps=steps, lr=lr, wd=1.0, evals=50, patience=5)      # donor = standard clean (wd=1) model on seed+500
        rows.append(dict(id=f'{block}_{task}_{arch}_{regime}_s{sd}' + ''.join(f'_{k}{v}' for k, v in model.items()) + (f"_{pen['kind']}{int(pen['lam']) if pen['lam'] >= 1 else pen['lam']}" if pen else ''), task=task, arch=arch, regime=regime, seed=sd, hp=hp))


MODSTEPS = dict(mod59=15000, mod97=12000, mod113=12000, comp60=25000, comp96=12000)
# A. clock vs pizza (alpha=1 = default attention = the clean runs of block B)
for t in ('mod59', 'mod97', 'mod113'): add('A', t, 'tf1', 'clean', S10, MODSTEPS[t], alpha=0.0)
# B. core regimes on the five modular tasks
for t, st in MODSTEPS.items():
    for reg in ('clean', 'scrambled', 'nograk', 'contaminated'): add('B', t, 'tf1', reg, S10, st)
    if t in ('mod59', 'mod97', 'mod113'): add('B', t, 'tf1', 'hybrid', S10, 30000, lr=3e-3)
# C. non-abelian groups
for t, a, lr, st in (('S5', 'mlp2', 3e-3, 30000), ('S5', 'tf2', 3e-3, 30000), ('A5', 'tf2', 1e-3, 20000), ('A5', 'mlp2', 3e-3, 20000), ('D12', 'mlp2', 1e-3, 20000)):
    for reg in ('clean', 'scrambled', 'nograk'): add('C', t, a, reg, S10, st, lr=lr)
for reg in ('clean', 'scrambled', 'nograk'): add('C', 'permcomp5', 'mlp2', reg, S10, 8000)
# D. other tasks
for t, st, kw in (('sorting', 8000, {}), ('copying', 8000, {}), ('dyck2', 20000, {}), ('parity', 8000, {}), ('symfree', 8000, {})):
    for a in ('mlp2', 'tf2'):
        lr = 3e-4 if (t == 'dyck2' and a == 'tf2') else 1e-3
        for reg in ('clean', 'scrambled'): add('D', t, a, reg, S10, st if not (t == 'dyck2' and a == 'mlp2') else 8000, lr=lr)
    for reg in ('clean', 'scrambled'): add('D', t, 'lstm', reg, S5, st)
# E. architecture sweep on mod59
for a, seeds, lr in (('mlp2', S10, 1e-3), ('tf2', S10, 1e-3), ('tf4', S10, 1e-3), ('tf8', S5, 5e-4), ('tf12', S5, 5e-4), ('lstm', S5, 1e-3), ('mamba', S5, 1e-3)):
    for reg in ('clean', 'scrambled'): add('E', 'mod59', a, reg, seeds, 15000 if a in ('tf8', 'tf12') else 10000, lr=lr)
# F. causal block (M5)
for t, a, st, lr in (('mod59', 'tf1', 15000, 1e-3), ('mod113', 'tf1', 12000, 1e-3), ('S5', 'mlp2', 30000, 3e-3), ('permcomp5', 'mlp2', 8000, 1e-3)):
    add('F', t, a, 'destroy', S10, st, lr=lr, wd=1.0, pen=DESTROY[10]); add('F', t, a, 'destroy', S10, st, lr=lr, wd=1.0, pen=DESTROY[30]); add('F', t, a, 'induce', S10, st, lr=lr, wd=0.0, pen=INDUCE); add('F', t, a, 'transplant', S10, st, lr=lr, wd=0.0, donor=True)
add('F', 'permcomp5', 'mlp2', 'nograk', S10, 8000)
ids = [r['id'] for r in rows]; assert len(ids) == len(set(ids)), 'duplicate ids'
with open('bench/phase2_grid.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, sort_keys=True) + '\n')
from collections import Counter
print(len(rows), dict(Counter(r['id'][0] for r in rows)))
