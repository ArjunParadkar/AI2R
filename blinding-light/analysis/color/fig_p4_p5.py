"""Deliverable 5: P4 and P5 against their baselines, every system on one page.
P4: AUROC of each score at separating held-out in-distribution inputs from OOD inputs.
P5: Spearman correlation of each predictor with the measured failure radius (the spec wants
    rho strongly NEGATIVE: high risk = small radius; softmax margin should be POSITIVE).
Only AUCs / correlations were stored, so this is a dot plot rather than ROC curves.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *

J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def main():
    setup()
    rows = []   # (label, {method: auc})
    for s in ('clean_s0', 'clean_s1', 'cleanctrl_s2'):
        d = J(f'runs/z113_{s}/static.json')
        if d:
            for k, v in d['P4']['sets'].items():
                rows.append((f'Z113 {s[-2:]} · {k}', dict(dg=v['auc_ignorance'], euclid=v['auc_euclid_knn'], softmax=v['auc_softmax'])))
    d = J('results/resnet_depth.json'); d0 = J('results/resnet.json')
    if d:
        for k, v in d['clean']['P4'].items():
            rows.append((f'ResNet-18 · {k}', dict(dg=v['auc_spec_dg'], tubular=v['auc_tubular'], euclid=v['auc_euclid_knn'], softmax=v['auc_softmax'])))
    elif d0:
        for k, v in d0['clean']['P4']['sets'].items():
            rows.append((f'ResNet-18 · {k}', dict(dg=v['auc_ignorance'], euclid=v['auc_euclid_knn'], softmax=v['auc_softmax'])))
    g = J('results/gpt2_explore.json')
    if g:
        for k, v in g['P4'].items():
            if isinstance(v, dict):
                rows.append((f'GPT-2 L6 · {k}', dict(dg=v['auc_spec_dg'], tubular=v['auc_tubular'], euclid=v['auc_euclid_knn'], softmax=v['auc_softmax'])))
    meth = [('dg', 'spec d_g (pre-registered)', BLUE, 'o'), ('tubular', 'tubular distance (post-hoc)', ORANGE, 's'),
            ('euclid', 'Euclidean kNN', AQUA, '^'), ('softmax', 'softmax confidence', YELLOW, 'D')]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 0.36 * len(rows) + 2.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v) in zip(ys, rows):
        ax.plot([min(v.values()), max(v.values())], [y, y], color=GRID, lw=1.5, zorder=1)
    for key, lab, col, mk in meth:
        pts = [(v[key], y) for y, (_, v) in zip(ys, rows) if key in v]
        if pts:
            ax.scatter(*zip(*pts), s=52, color=col, marker=mk, label=lab, zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=':'); ax.set_yticks(ys, [r[0] for r in rows], fontsize=8); ax.set_xlim(0.25, 1.02)
    ax.set_xlabel('AUROC, in-distribution vs out-of-distribution (0.5 = chance)'); ax.set_title('P4  what it does not know')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8); ax.grid(axis='y', visible=False)
    # ---- P5
    r5 = []
    for s in ('clean_s0', 'clean_s1', 'cleanctrl_s2'):
        d = J(f'runs/z113_{s}/static.json')
        if d:
            r5.append((f'Z113 {s[-2:]}', d['P5']['spearman_rho'], d['P5']['spearman_softmax_margin'], d['P5'].get('null_spearman_shuffled_rho')))
    if d0:
        r5.append(('ResNet-18 (curv. dim 4, pre-fix)', d0['clean']['P5']['spearman_rho'], d0['clean']['P5']['spearman_softmax_margin'], d0['clean']['P5'].get('null_shuffled')))
    if g and 'P5' in g:
        r5.append(('GPT-2 L6', g['P5']['spearman_rho'][0], g['P5']['spearman_softmax_margin'][0], g['P5']['null_shuffled_rho'][0]))
    y5 = np.arange(len(r5))[::-1]
    ax2.axvspan(-1, -0.5, color=GRID, alpha=0.6, zorder=0); ax2.annotate('pass region for ρ', (-0.98, len(r5) - 0.55), fontsize=8, color=INK2)
    ax2.scatter([r[1] for r in r5], y5, s=60, color=BLUE, marker='o', label='ρ = ‖II‖·Lip (wanted: < −0.5)', zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax2.scatter([r[2] for r in r5], y5, s=60, color=YELLOW, marker='D', label='softmax margin (baseline)', zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax2.scatter([r[3] for r in r5], y5, s=40, color=INK2, marker='x', label='shuffled-ρ null', zorder=3)
    ax2.axvline(0, color=INK2, lw=0.8); ax2.set_yticks(y5, [r[0] for r in r5], fontsize=8); ax2.set_xlim(-1, 1); ax2.set_ylim(-0.6, len(r5) - 0.3)
    ax2.set_xlabel('Spearman correlation with measured failure radius'); ax2.set_title('P5  where it will fail')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8); ax2.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig('figures/color/p4_p5.png'); print('figures/color/p4_p5.png')


if __name__ == '__main__':
    main()
