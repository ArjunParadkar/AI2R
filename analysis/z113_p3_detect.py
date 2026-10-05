"""P3 detection across the checkpoints of already-trained runs: when is a spurious class first
detected, against the ground-truth onset (first checkpoint where ablating the hint moves
accuracy on the leaked rows by > 5 points), and how many false positives on clean models.

tau (condition ii) is calibrated on the two frac=0.3 clean seeds and then applied, unchanged,
to the held-out clean control and to the shortcut models.
"""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.z113_common import checkpoints, load
from analysis.z113_interrupt import spurious_report
from systems.z113 import build_data

RUNS = dict(clean_s0=('clean', 0, 0.3), clean_s1=('clean', 1, 0.3), cleanctrl_s2=('clean', 2, 0.5),
            shortcut_s0=('shortcut', 0, 0.5), shortcut_s1=('shortcut', 1, 0.5))


def scan(tag, every=1):
    mode, seed, frac = RUNS[tag]
    # the probe needs training (a,b) pairs only; the hint is swept by the probe itself
    d = build_data(seed, mode, frac=frac)
    cks, steps = checkpoints(f'runs/z113_{tag}')
    out = []
    for ck, st in list(zip(cks, steps))[::every]:
        m = load(ck, seed=seed)
        sp = spurious_report(m, d, tau=None)
        for c in sp:
            c.pop('_geom', None)
        out.append(dict(step=int(st), classes=sp))
    return out


def main():
    res = {t: scan(t) for t in RUNS}
    # tau: the largest sensitivity any hint-tracking class reaches on the calibration clean seeds
    cal = [c['sensitivity'] for t in ('clean_s0', 'clean_s1') for r in res[t] for c in r['classes']
           if c['corr_z'] > 0.5 and c['life_ratio'] > 3 and c['var'] > 0.02]
    tau = 2.0 * max(cal) if cal else 0.0
    summary = dict(tau=tau, n_calibration_candidates=len(cal))
    for t, recs in res.items():
        det = [dict(step=r['step'], n=sum(1 for c in r['classes'] if c['corr_z'] > 0.5 and c['life_ratio'] > 3
                                          and c['var'] > 0.02 and c['sensitivity'] > tau)) for r in recs]
        first = next((x['step'] for x in det if x['n'] > 0), None)
        summary[t] = dict(first_detection_step=first, n_ckpts_flagged=sum(1 for x in det if x['n'] > 0), n_ckpts=len(det))
        if t.startswith('shortcut'):
            h = json.load(open(f'runs/z113_{t}/history.json'))['history']
            summary[t]['ground_truth_onset_step'] = next((r['step'] for r in h if r.get('shortcut_reliance', 0) > 0.05), None)
        print(t, summary[t], flush=True)
    json.dump(dict(summary=summary, scans=res), open('results/p3_detect.json', 'w'), indent=1, default=float)
    print('tau', tau)


if __name__ == '__main__':
    main()
