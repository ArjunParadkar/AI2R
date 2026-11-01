"""Run-2 figures owned by the main session: alignment vs H^1 silence, vineyards, m_plateau vs old m.
300 dpi, black-and-white safe: series are separated by line style and marker, never by hue alone."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
K, G1, G2 = '#000000', '#555555', '#999999'


def fig_alignment():
    d = json.load(open('results/run2/blockD12.json')); v = d['_verdict']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, s in zip(axes, (0, 1)):
        R = d[f'r2_reencoded_s{s}']; st = np.array([r['step'] for r in R]) / 1000
        ax.plot(st, [r['direct_nmi'] for r in R], color=K, ls='-', marker='o', ms=5, label='direct NMI(prediction, hint)')
        ax.plot(st, [r['A_ami'] for r in R], color=G1, ls='--', marker='s', ms=4, label='Mapper alignment, chance-adjusted (AMI)')
        ax.plot(st, [r['A'] for r in R], color=G2, ls=':', marker='^', ms=4, label='Mapper alignment, raw NMI (pre-registered)')
        ax.plot(st, [r.get('reliance', np.nan) for r in R], color=K, ls='-.', lw=1.2, label='ablation reliance (ground truth)')
        ax.axhline(v['threshold_direct'], color=K, lw=0.8); ax.annotate(f"direct threshold {v['threshold_direct']:.2f}", (0.05, v['threshold_direct'] + 0.015), fontsize=8)
        for r in R:
            if r.get('h1_spurious') == 0 and r.get('reliance', 0) > 0.9:
                ax.axvspan(r['step'] / 1000 - 0.22, r['step'] / 1000 + 0.22, facecolor='none', edgecolor=K, hatch='///', lw=0.0, alpha=0.5)
        ax.set_title(f're-encoded model, seed {s}'); ax.set_xlabel('training step (thousands)'); ax.set_ylim(-0.03, 1.12)
        ax.annotate('hatched: H¹ detector reports 0 classes while reliance > 0.9', (0.02, 1.06), fontsize=8, color=INK2)
        ax.annotate(f"pre-registered raw-NMI threshold = {v['threshold']:.2f} (unreachable)", (0.02, 0.0), fontsize=8, color=INK2)
    axes[0].set_ylabel('statistic (all in [0, 1])'); h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.03)); fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig('figures/run2_alignment_vs_h1.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_vineyard():
    d = json.load(open('results/run2/blockF.json')); names = [k for k in d if not k.startswith('_')]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.8), sharey=True); ls = ['-', '--', '-.', ':', (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))]
    for ax, n in zip(axes, names):
        for i, vn in enumerate(sorted(d[n]['vines'], key=lambda v: v['freq'])):
            y = np.array(vn['nlife']); ax.plot(np.arange(len(y)), y, color=K if vn['survives'] else G2, ls=ls[i % len(ls)], lw=1.4 if vn['survives'] else 0.9, label=f"k={vn['freq']}")
            if vn['onset'] is not None:
                ax.plot(vn['onset'], y[vn['onset']], marker='o', ms=5, color=K, mfc='white')
        ax.axvline(d[n]['grok_ckpt'], color=K, lw=0.8); ax.annotate('grok', (d[n]['grok_ckpt'], ax.get_ylim()[1] * 0.02), fontsize=8, rotation=90, va='bottom', ha='right')
        ax.set_title(f"{n.replace('z113_', '')} ({d[n]['status']})", fontsize=9); ax.set_xlabel('checkpoint'); ax.legend(fontsize=6.5, loc='upper left', ncol=2)
    axes[0].set_ylabel('normalised H¹ lifetime of the frequency circle')
    fig.suptitle('Formation vineyard: one vine per frequency circle (symmetry-informed). Open circle = onset; grey = does not survive to the end.', fontsize=9, y=1.0)
    fig.tight_layout(); fig.savefig('figures/run2_vineyards.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_mplateau():
    d = json.load(open('results/run2/blockG.json')); ks = [5, 10, 20, 40, 80, 160]
    show = [('z113_clean_s1', 'Z113 grokked', '-', 'o'), ('z113_scrambled_s0', 'Z113 memoriser', '--', 'o'), ('ring_s1', 'ring RNN', '-', 's'),
            ('resnet_clean_penultimate', 'ResNet clean, penultimate', '-', '^'), ('resnet_scrambled_penultimate', 'ResNet scrambled, penultimate', '--', '^'),
            ('resnet_clean_layer3', 'ResNet clean, layer3', ':', '^'), ('gpt2_L6', 'GPT-2 layer 6', '-.', 'D')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for key, lab, l, mk in show:
        a1.plot(ks[:len(d[key]['id_curve'])], d[key]['id_curve'], color=K, ls=l, marker=mk, ms=5, mfc='white', lw=1.3, label=lab)
    a1.set_xscale('log'); a1.set_xticks(ks, [str(k) for k in ks]); a1.set_xlabel('neighbourhood size k'); a1.set_ylabel('estimated intrinsic dimension'); a1.legend(fontsize=7.5)
    a1.set_title('ID(k): plateau, decay, growth')
    keys = [k for k in d if not k.startswith('_')]
    a2.scatter([d[k]['old_m']['default'] for k in keys], [d[k]['m_plateau'] for k in keys], s=34, facecolor='white', edgecolor=K)
    for k in ('z113_clean_s1', 'z113_clean_s0', 'gpt2_L6', 'resnet_clean_penultimate', 'ring_s1', 'resnet_clean_layer3'):
        a2.annotate(k.replace('z113_', 'Z113 ').replace('resnet_clean_', 'ResNet ').replace('_', ' '), (d[k]['old_m']['default'], d[k]['m_plateau']), xytext=(5, 4), textcoords='offset points', fontsize=7.5)
    a2.set_xlabel('old spectral-gap m'); a2.set_ylabel('m_plateau'); a2.set_xlim(-0.05, 1.05); a2.set_ylim(-0.05, 1.05); a2.set_title('neither scalar orders the systems as expected')
    fig.tight_layout(); fig.savefig('figures/run2_mplateau.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup(); fig_alignment(); fig_vineyard(); fig_mplateau(); print('ok')
