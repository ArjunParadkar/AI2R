"""The harder P3 null: models trained with a hint token that is PRESENT (as often as in the
planted-shortcut runs) but uninformative.  Condition (i) can now fire -- the hint is a trained
input and may well be laid out on a circle -- so zero detections here must come from condition
(ii), output dependence.  tau: session 1 fixed tau = 2 x the largest sensitivity on the clean
calibration seeds; on those no class passed (i), so tau = 0 and (ii) was vacuous.  Here we
report the sensitivities themselves next to the shortcut models', and the ablation ground truth.
"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load
from analysis.z113_interrupt import spurious_report
from systems.z113 import build_data, NULL


def main():
    out = {}
    for tag, mode, seed in (('decoy_s0', 'decoy', 0), ('decoy_s1', 'decoy', 1), ('shortcut_s0', 'shortcut', 0), ('shortcut_s1', 'shortcut', 1)):
        d = build_data(seed, mode, frac=0.5)
        cks, steps = checkpoints(f'runs/z113_{tag}')
        rows = []
        for ck, st in list(zip(cks, steps))[::2]:
            m = load(ck, seed=seed)
            sp = spurious_report(m, d, tau=None)
            for c in sp:
                c.pop('_geom', None)
            # ground truth: does removing the hint change the training-set predictions?
            x = torch.as_tensor(d['train_toks'], device='cuda'); xa = x.clone(); xa[:, 2] = NULL
            with torch.no_grad():
                changed = float((m(x).argmax(-1) != m(xa).argmax(-1)).float().mean())
            cand = [c for c in sp if c['corr_z'] > 0.5 and c['life_ratio'] > 3 and c['var'] > 0.02]
            rows.append(dict(step=int(st), n_hint_circles=len(cand), max_sensitivity=max([c['sensitivity'] for c in cand], default=0.0),
                             pred_changed_by_ablation=changed))
        out[tag] = rows
        print(tag, 'ckpts with a hint-tracking circle (cond. i):', sum(r['n_hint_circles'] > 0 for r in rows), '/', len(rows),
              '| max sensitivity %.3g' % max(r['max_sensitivity'] for r in rows),
              '| final ablation effect %.3f' % rows[-1]['pred_changed_by_ablation'], flush=True)
    json.dump(out, open('results/p3_decoy.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
