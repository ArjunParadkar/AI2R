"""Ring RNN, P7: when is the circle born, against when integration error drops?

At every checkpoint: g-geodesic Rips on 600 states (unsupervised; heading is used only to score).
'Present' = one dominant class (lifetime ratio > 3) whose coordinate tracks heading at r > 0.9.
Birth = first checkpoint from which the circle is present to the end.  Error drop = first
checkpoint with mean angular error < 0.3 rad (the threshold session 1 fixed before looking).
"""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.ring_rnn import RingRNN
from whiteboxtopo.extract import jacobians
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.problems.core import geodesic_distance_matrix
from analysis.ring_pullback import cloud, top_circle


def main():
    out = {}
    for seed in (0, 1, 2):
        run = f'runs/ring_s{seed}'
        cks = sorted(glob.glob(f'{run}/ckpt_*.pt')); hist = json.load(open(f'{run}/history.json'))['history']
        rows = []
        for ck, h in zip(cks, hist):
            m = RingRNN(seed=seed).cuda(); m.load_state_dict(torch.load(ck)); m.eval()
            X, th, _ = cloud(m, 600, seed=123)
            J = jacobians(m.psi, X).astype(np.float32)
            c = top_circle(geodesic_distance_matrix(fit_manifold(X, J, metric='g')), th, True)
            rows.append(dict(step=h['step'], ang_err=h['ang_err_rad'], r=c['r'], ratio=c['ratio'],
                             present=bool(c['r'] > 0.9 and c['ratio'] > 3)))
        ok = [r['present'] for r in rows]
        birth = next((i for i in range(len(ok)) if all(ok[i:])), None)
        errs = np.array([r['ang_err'] for r in rows]); drop = int(np.argmax(errs < 0.3)) if (errs < 0.3).any() else None
        out[f's{seed}'] = dict(sweep=rows, birth_ckpt=birth, err_drop_ckpt=drop)
        print(f'[ring s{seed}] birth ckpt {birth}  err-drop ckpt {drop}', flush=True)
        print('    r    :', ' '.join(f"{r['r']:.2f}" for r in rows), flush=True)
        print('    ratio:', ' '.join(f"{min(r['ratio'], 99):4.1f}" for r in rows), flush=True)
        print('    err  :', ' '.join(f"{r['ang_err']:.2f}" for r in rows), flush=True)
    json.dump(out, open('results/ring_birth.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
