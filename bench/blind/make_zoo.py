"""Phase 5 (INTERNAL self-blinded study; NOT the external collaborator study, which needs a human at MIT).  Authors: Arjun Paradkar and Reva Paradkar.
make_zoo.py builds a mixed zoo with a hidden label per model.  Everything a predictor may use is in zoo/public (weights, task name, arch, seed, the TRAINING inputs and the labels
actually used for training, and the held-out INPUTS without labels).  The answer key (regime, alpha, true test accuracy per input) is written to ~/.wb_sealed/key_<sha>.json (mode 0400) and
its sha256 is committed in the prediction file; the key is not opened until predictions are committed and tagged.  python bench/blind/make_zoo.py build|seal
Seeds 7001-7999 are used nowhere else."""
import sys, os, json, hashlib, random, stat
import numpy as np, torch
sys.path.insert(0, '.')
from bench import zoo
from whiteboxtopo.numerics import repro

PUB = 'results/blind/public'; SEAL = os.path.expanduser('~/.wb_sealed')
SPECS = []                      # (task, regime, alpha)
for task in ('mod59', 'mod97'):
    for regime, alpha in (('clean', 1.0), ('clean', 0.0), ('scrambled', 1.0), ('nograk', 1.0), ('hybrid', 1.0), ('contaminated', 1.0)):
        for rep in range(2): SPECS.append((task, regime, alpha))
STEPS = dict(mod59=15000, mod97=12000)


def hp_for(task, regime, alpha):
    wd = 0.0 if regime in ('nograk', 'contaminated') else 1.0
    return dict(steps=STEPS[task] if regime != 'hybrid' else 30000, lr=3e-3 if regime == 'hybrid' else 1e-3, wd=wd, evals=50, patience=5, **({'model': dict(alpha=alpha)} if alpha != 1.0 else {}))


def build():
    os.makedirs(PUB, exist_ok=True); os.makedirs(SEAL, exist_ok=True); rng = random.Random(20261005); order = list(range(len(SPECS))); rng.shuffle(order); key = {}
    for new_id, k in enumerate(order):
        task, regime, alpha = SPECS[k]; seed = 7001 + new_id; mid = f'm{new_id:02d}'
        if os.path.exists(f'{PUB}/{mid}.pt'): key[mid] = json.load(open(f'{PUB}/{mid}.meta.json'))['_key']; continue
        repro.set_determinism(seed); t = zoo.get_task(task); data = zoo.build_data(t, seed, regime); hp = hp_for(task, regime, alpha)
        model, mid_snap, traj, info = zoo.train(t, 'tf1', regime, seed, hp, data)
        torch.save({k2: v.detach().cpu() for k2, v in model.state_dict().items()}, f'{PUB}/{mid}.pt')
        if mid_snap is not None: torch.save({k2: v.cpu() for k2, v in mid_snap['state'].items()}, f'{PUB}/{mid}_midsnap.pt')
        np.savez(f'{PUB}/{mid}_data.npz', Xtr=data['Xtr'], Ytr=data['Ytr'], Xte=data['Xte'])      # held-out INPUTS only, no labels
        meta = dict(id=mid, task=task, arch='tf1', seed=seed, n_params=sum(p.numel() for p in model.parameters()), has_midsnap=mid_snap is not None)
        kk = dict(regime=regime, alpha=alpha, test_acc=float(zoo.acc_of(zoo.predict(model, torch.as_tensor(data['Xte'], device=zoo.DEV)), torch.as_tensor(data['Yte'], device=zoo.DEV))),
                  train_acc=float(zoo.acc_of(zoo.predict(model, torch.as_tensor(data['Xtr'], device=zoo.DEV)), torch.as_tensor(data['Ytr'], device=zoo.DEV))), Yte=data['Yte'].ravel().tolist(),
                  pred_te=zoo.predict(model, torch.as_tensor(data['Xte'], device=zoo.DEV)).argmax(-1).cpu().numpy().ravel().tolist(), midsnap_test_acc=None if mid_snap is None else mid_snap['test_acc'])
        key[mid] = kk; meta['_key'] = kk; json.dump({k3: v for k3, v in meta.items() if k3 != '_key'} | {'_key': kk}, open(f'{PUB}/{mid}.meta.json', 'w')); print(mid, task, regime, alpha, flush=True)
    return key


def seal(key):
    blob = json.dumps(key, sort_keys=True).encode(); sha = hashlib.sha256(blob).hexdigest(); p = f'{SEAL}/key_{sha[:16]}.json'
    open(p, 'wb').write(blob); os.chmod(p, stat.S_IRUSR)
    for f in os.listdir(PUB):                                  # strip the key out of the public meta files
        if f.endswith('.meta.json'):
            m = json.load(open(f'{PUB}/{f}')); m.pop('_key', None); json.dump(m, open(f'{PUB}/{f}', 'w'))
    json.dump(dict(sha256=sha, n=len(key)), open('results/blind/key_commitment.json', 'w')); print('sealed', sha)


if __name__ == '__main__':
    if sys.argv[1] == 'build': k = build(); seal(k)
