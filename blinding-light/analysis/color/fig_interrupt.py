"""Deliverable 4: the interrupt figure.  Accuracy on the DECORRELATED test set (hint uniform,
independent of the answer) through training, for each intervention, both seeds.  Parsed from the
run logs so it can be drawn while runs are still going.  Vertical marks: ground-truth onset
(first checkpoint where ablating the hint moves leaked-row accuracy > 5 points) and detection.
"""
import sys, os, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *

PAT = re.compile(r'\[(\w+) s(\d)\]\s+(\d+) test ([\d.]+) dec ([\d.]+) worst ([\d.]+) reliance ([-\d.]+) spurious (\d+) lam ([\d.]+)')
MODES = [('interrupt', 'logs/p3_interrupt_s{}.log', 'interrupt: input-level KL consistency (needs z)', BLUE),
         ('structure', 'logs/p3_structure_s{}.log', 'interrupt: along the discovered circle (no z)', ORANGE),
         ('spare', 'logs/p3b_spare_s{}.log', 'SPARE-style reweighting', AQUA),
         ('none', None, 'no intervention', YELLOW)]


def parse(path):
    rows = []
    if path and os.path.exists(path):
        for line in open(path):
            m = PAT.search(line)
            if m:
                rows.append(dict(step=int(m[3]), test=float(m[4]), dec=float(m[5]), worst=float(m[6]), reliance=float(m[7]),
                                 n_spurious=int(m[8]), lam=float(m[9])))
    return rows


def main():
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    summary = {}
    for ax, seed in zip(axes[:2], (0, 1)):
        for mode, pat, lab, col in MODES:
            if mode == 'none':
                h = json.load(open(f'runs/z113_shortcut_s{seed}/history.json'))['history']
                rows = [dict(step=r['step'], dec=r['dec_acc'], reliance=r.get('shortcut_reliance', 0)) for r in h if r.get('dec_acc') is not None]
            else:
                rows = parse(pat.format(seed))
            if not rows:
                continue
            st = np.array([r['step'] for r in rows]) / 1000; dec = np.array([r['dec'] for r in rows])
            ax.plot(st, dec, color=col, label=lab)
            if mode == 'interrupt':
                end_label(ax, st[-1], dec[-1], f'{dec[-1]:.2f}', dy=(8 if dec[-1] < 0.5 else 0))
            summary[f'{mode}_s{seed}'] = dict(last_step=rows[-1]['step'], dec_acc=float(dec[-1]), worst=rows[-1].get('worst'),
                                              reliance=rows[-1].get('reliance'), max_lam=max(r.get('lam', 0) for r in rows),
                                              first_detection=next((r['step'] for r in rows if r.get('n_spurious', 0) > 0), None))
            if mode == 'interrupt':
                det = summary[f'{mode}_s{seed}']['first_detection']
                onset = next((r['step'] for r in rows if r['reliance'] > 0.05), None)
                if onset is not None:
                    ax.axvline(onset / 1000, color=INK2, lw=0.8, ls=':'); ax.annotate('onset', (onset / 1000, 1.02), fontsize=8, color=INK2, ha='center')
                if det is not None and det != onset:
                    ax.axvline(det / 1000, color=INK2, lw=0.8, ls='--'); ax.annotate('detected', (det / 1000, 0.94), fontsize=8, color=INK2, ha='left')
        ax.annotate('structure-derived penalty, SPARE and no intervention: 0.01–0.02', (12.5, 0.06), ha='center', fontsize=8, color=INK2)
        ax.set_title(f'planted shortcut, seed {seed}'); ax.set_xlabel('training step (thousands)'); ax.set_ylim(-0.03, 1.08)
    # third panel: the pre-registered interrupt on every seed, plus the post-hoc gentle schedule on the seed that collapsed
    ax = axes[2]; seedcol = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    for sd in range(5):
        rows = parse(f'logs/p3_interrupt_s{sd}.log')
        if rows:
            st = np.array([r['step'] for r in rows]) / 1000; dec = np.array([r['dec'] for r in rows]); keep = st <= 6
            ax.plot(st[keep], dec[keep], color=seedcol[sd], label=f'seed {sd}')
            summary[f'interrupt_s{sd}'] = dict(last_step=rows[-1]['step'], dec_acc=float(dec[-1]), worst=rows[-1]['worst'], reliance=rows[-1]['reliance'],
                                               max_lam=max(r['lam'] for r in rows), first_detection=next((r['step'] for r in rows if r['n_spurious'] > 0), None),
                                               onset=next((r['step'] for r in rows if r['reliance'] > 0.05), None))
    rows = parse('logs/p3_interrupt_gentle_s1.log')
    if rows:
        st = np.array([r['step'] for r in rows]) / 1000; dec = np.array([r['dec'] for r in rows])
        ax.plot(st, dec, color=seedcol[1], ls='--', label='seed 1, gentle schedule (post-hoc)')
        summary['interrupt_gentle_s1'] = dict(last_step=rows[-1]['step'], dec_acc=float(dec[-1]), reliance=rows[-1]['reliance'], max_lam=max(r['lam'] for r in rows))
    ax.annotate('seeds 0, 2, 3, 4: 1.00', (6, 1.0), xytext=(-4, -12), textcoords='offset points', ha='right', fontsize=9, color=INK)
    ax.annotate('seed 1 (both schedules): 0.01', (6, 0.01), xytext=(-4, 8), textcoords='offset points', ha='right', fontsize=9, color=INK)
    ax.set_title('pre-registered interrupt, five seeds (first 6k steps)'); ax.set_xlabel('training step (thousands)'); ax.legend(loc='center right', fontsize=8)
    axes[0].set_ylabel('accuracy on the decorrelated test set'); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.36, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig('figures/color/interrupt.png')
    json.dump(summary, open('results/p3_interrupt_summary.json', 'w'), indent=1); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
