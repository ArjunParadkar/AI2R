"""Deliverable 2: train accuracy, test accuracy and G(t) on ONE axis, frequency births marked."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *

RUNS = [('z113_clean_s0', 'clean, seed 0'), ('z113_clean_s1', 'clean, seed 1'),
        ('z113_cleanctrl_s2', 'clean control (50% train), seed 2'), ('z113_scrambled_s0', 'scrambled labels (null)')]


def births(R):
    """First checkpoint from which a frequency's certified circle is present to the end."""
    fr = sorted({c['freq'] for r in R for c in r['circles'] if c['present']})
    out = {}
    for k in fr:
        ok = [any(c['freq'] == k and c['present'] for c in r['circles']) for r in R]
        b = next((i for i in range(len(ok)) if all(ok[i:])), None)
        if b is not None:
            out[k] = b
    return out


def main():
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharey=True)
    for ax, (run, title) in zip(axes.ravel(), RUNS):
        p = f'runs/{run}/sweep.json'
        if not os.path.exists(p):
            ax.set_visible(False); continue
        R = json.load(open(p))['records']
        if 'G_interior' not in R[0]:
            ax.set_title(title + ' -- re-sweep pending'); continue
        st = np.array([r['step'] for r in R]) / 1000
        series = [('train_acc', 'train accuracy', BLUE, '-'), ('test_acc', 'test accuracy', ORANGE, '-'),
                  ('G', 'G (pre-registered)', AQUA, '-'), ('G_interior', 'G interior (post-hoc)', YELLOW, '-')]
        ends = []
        for key, lab, col, ls in series:
            y = np.array([np.nan if r[key] is None else r[key] for r in R], float)
            ax.plot(st, y, color=col, ls=ls, label=lab); ends.append((y[-1], lab))
        b = births(R)
        for k, i in b.items():
            ax.axvline(st[i], color=INK2, lw=0.8, ls=':')
        if b:
            ax.annotate('circle births: ' + ', '.join(f'k={k}' for k in sorted(b, key=b.get)), (0.99, 0.04),
                        xycoords='axes fraction', ha='right', fontsize=8, color=INK2)
        ax.set_title(title); ax.set_ylim(-0.03, 1.05); ax.set_xlabel('training step (thousands)')
    axes[0, 0].set_ylabel('fraction'); axes[1, 0].set_ylabel('fraction')
    h, l = axes[0, 1].get_legend_handles_labels() if axes[0, 1].get_visible() else axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Geodesic closure of the training representation across grokking (Z113). Dotted lines: birth of each certified frequency circle.',
                 y=0.04, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig('figures/color/grokking.png'); print('figures/color/grokking.png')


if __name__ == '__main__':
    main()
