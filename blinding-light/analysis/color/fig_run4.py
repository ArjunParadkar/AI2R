"""Run-4 figures.  300 dpi, black-and-white safe (marker + line style + hatch carry identity)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
K, G1, G2 = '#1f4e79', '#e0662f', '#6fb1e3'
J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def fig_p6():
    Z, R = J('results/run4/z113_finals.json') or {}, J('results/run4/resnet.json') or {}
    panels = [('P6.1 concentration C', [(k, v['P6_1']['C']) for k, v in {**Z, **R}.items() if 'clean' in k or 'scrambled' in k], None),
              ('P6.2 max slot share (Z113)', [(k, max(v['slot_share'])) for k, v in Z.items() if 'clean' in k or 'scrambled' in k], 0.5),
              ('P6.3 q(swap) (Z113)', [(k, v['q_swap']) for k, v in Z.items() if 'clean' in k or 'scrambled' in k], None),
              ('P6.3 shift consistency (ResNet)', [(k, v['P6_3']['shift_consistency']) for k, v in R.items()], None),
              ('P6.4 connection-scale slope', [(k, v['connection']['slope']) for k, v in Z.items() if 'connection' in v] + [(k, v['P6_4']['slope']) for k, v in R.items()], None),
              ('P6.5 full 113-cycle found (Z113)', [(k, float(v['group']['full_cycle'])) for k, v in Z.items() if 'clean' in k or 'scrambled' in k], None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    for ax, (title, items, _) in zip(axes.ravel(), panels):
        items = sorted(items, key=lambda t: ('scrambled' in t[0], t[0])); names = [k.replace('r4_', '').replace('z113_', 'Z ').replace('resnet_', 'R ').replace('scrambled_decoy', 'mem').replace('scrambled', 'mem') for k, _ in items]
        for i, (k, v) in enumerate(items):
            mem = 'scrambled' in k; ax.barh(i, v, color='#fff1b8' if mem else G1, edgecolor=K, hatch='///' if mem else None, height=0.62)
            ax.text(v, i, f' {v:.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        ax.set_yticks(range(len(items)), names, fontsize=8); ax.invert_yaxis(); ax.axvline(0, color=K, lw=0.8); ax.set_title(title, fontsize=10); ax.grid(axis='y', visible=False)
    fig.suptitle('Five instruments on matched pairs. Solid grey = generaliser; hatched = true memoriser (train ≥ 99%, test at chance).', fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run4_p6_instruments.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_tower():
    Z, R = J('results/run4/z113_finals.json') or {}, J('results/run4/resnet.json') or {}
    show = [k for k in ('r4_z113_clean_s5', 'r4_z113_decoy_s5', 'r4_z113_scrambled_decoy_s5', 'r4_z113_shortcut_s5') if k in Z] + [k for k in ('r4_resnet_clean_s1', 'r4_resnet_scrambled_s1') if k in R]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.2))
    for ax, k in zip(np.atleast_1d(axes), show):
        T = Z[k]['tower'] if k in Z else R[k]['tower']; rows = list(T); cols = [c for c in T[rows[0]] if c != 'first_null_layer']
        M = np.array([[T[r][c] for c in cols] for r in rows]); ax.imshow(np.clip(M, 0, 1), cmap='viridis', vmin=0, vmax=1, aspect='auto')
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, f'{M[i, j]:.2f}' + ('*' if M[i, j] < 0.05 else ''), ha='center', va='center', fontsize=7.5, color='#fff1b8' if M[i, j] > 0.55 else 'black')
        ax.set_xticks(range(len(cols)), [c.replace('resid_', '').replace('penultimate', 'penult') for c in cols], fontsize=7, rotation=30); ax.set_yticks(range(len(rows)), [r.replace('_', ' ') for r in rows], fontsize=8)
        ax.set_title(k.replace('r4_', '').replace('z113_', 'Z113 ').replace('resnet_', 'ResNet ').replace('scrambled_decoy', 'memoriser').replace('scrambled', 'memoriser'), fontsize=9); ax.grid(False)
    fig.suptitle('Tower of quotients: q(T) by layer (0 = the layer is blind to T; * = below 0.05).  Z113: linearised pullback; ResNet: Euclidean.', fontsize=9, y=0.0)
    fig.savefig('figures/color/run4_tower.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_formation():
    runs = [k for k in ['r4_z113_clean_s%d' % s for s in (5, 6, 7, 8)] + ['r4_z113_decoy_s5', 'r4_z113_decoy_s6', 'r4_z113_scrambled_decoy_s5'] if os.path.exists(f'results/run4/sweep_{k}.json')]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.5 * len(runs), 3.7), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), runs):
        S = J(f'results/run4/sweep_{k}.json'); st = np.array([r['step'] for r in S]) / 1000
        ax.plot(st, [r['test_acc'] or 0 for r in S], color=K, ls='-', lw=1.6, label='test accuracy'); ax.plot(st, [r['train_acc'] or 0 for r in S], color=G2, ls='-', lw=1.0, label='train accuracy')
        ax.plot(st, [float(r['full_cycle']) for r in S], color=K, ls=':', marker='o', ms=4, mfc='#fff1b8', label='113-cycle generator found')
        ax.plot(st, [min(r['q_swap'], 1) for r in S], color=G1, ls='--', marker='s', ms=3, label='q(swap)'); ax.plot(st, [r['hint_share'] for r in S], color=K, ls='-.', marker='^', ms=3, mfc='#fff1b8', label='hint-slot share of trace(ḡ)')
        ax.set_title(k.replace('r4_z113_', '').replace('scrambled_decoy', 'memoriser'), fontsize=9); ax.set_xlabel('step (thousands)'); ax.set_ylim(-0.04, 1.08)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.savefig('figures/color/run4_formation.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_dominance():
    Z = J('results/run4/z113_finals.json') or {}
    if not Z:
        return
    ks = sorted(Z, key=lambda k: (Z[k]['mode'], k)); fig, ax = plt.subplots(figsize=(10, 0.42 * len(ks) + 1.4)); left = np.zeros(len(ks))
    for j, (nm, hat, col) in enumerate((('slot a', None, G1), ('slot b', '...', '#fff1b8'), ('hint slot', '///', '#fff1b8'))):
        v = np.array([Z[k]['slot_share'][j] for k in ks]); ax.barh(range(len(ks)), v, left=left, color=col, edgecolor=K, hatch=hat, height=0.65, label=nm); left += v
    ax.axvline(0.8, color=K, lw=1, ls='--'); ax.set_yticks(range(len(ks)), [k.replace('r4_z113_', '').replace('scrambled_decoy', 'memoriser') for k in ks], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('share of trace(ḡ) per input slot (decorrelated probe); rule fires when one slot exceeds 0.8'); ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8); ax.grid(axis='y', visible=False)
    fig.savefig('figures/color/run4_dominance.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_interrupt():
    fig, ax = plt.subplots(figsize=(7.5, 4)); any_ = False; ls = ['-', '--', '-.', ':', (0, (5, 1))]
    for s in range(5):
        H = J(f'runs/r4_interrupt_s{s}/history.json')
        if H:
            any_ = True; h = H['history']; ax.plot([x['step'] / 1000 for x in h], [x['dec_acc'] for x in h], color=K, ls=ls[s], marker='os^Dv'[s], ms=3, mfc='#fff1b8', label=f'seed {s}')
    if any_:
        ax.axvline(6, color=G1, lw=0.8); ax.axhline(0.98, color=G1, lw=0.8, ls=':'); ax.set_xlabel('step (thousands)'); ax.set_ylabel('accuracy on the decorrelated test'); ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=8)
        ax.set_title('P3.2: functional-floor trigger, presence-and-value KL penalty, λ = 3 once', fontsize=10); fig.savefig('figures/color/run4_interrupt.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_p6, fig_tower, fig_formation, fig_dominance, fig_interrupt):
        try:
            f(); print('ok', f.__name__)
        except Exception as e:
            print('FAILED', f.__name__, type(e).__name__, str(e)[:160])
