"""Run-5 figures.  300 dpi, black-and-white safe (hatch, marker and line style carry identity; thresholds are dashed lines)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
K, G1, G2 = '#000000', '#555555', '#aaaaaa'
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
nm = lambda k: k.replace('r5_', '').replace('z113_', 'Z ').replace('resnet_', 'R ').replace('scrambled_decoy', 'mem').replace('scrambled', 'mem').replace('cleanaug', 'clean+aug')
LAY = ('embed', 'resid_mid', 'resid_post', 'logits')


def fig_instruments():
    Z, R = J('results/run5/z113_finals.json') or {}, J('results/run5/resnet.json') or {}; zz = {k: v for k, v in Z.items() if v['mode'] in ('clean', 'scrambled_decoy')}
    panels = [('P6.a  q(swap) at resid_mid (Z113)', [(k, v['q_swap']) for k, v in zz.items()], (0.10, 0.30), [(k, v['q_swap_untrained']) for k, v in zz.items()]),
              ('P6.b  shift consistency ±2 px (ResNet)', [(k, v['P6_b']['shift_consistency']) for k, v in R.items()], (0.75, 0.35), [(k, v['P6_b']['untrained']) for k, v in R.items()]),
              ('P6.c  connection-scale depth slope', [(k, v['connection']['slope']) for k, v in zz.items()] + [(k, v['P6_c']['slope']) for k, v in R.items() if 'P6_c' in v], (-0.35, -0.30),
               [(k, v['connection']['null_mean']) for k, v in zz.items()] + [(k, v['P6_c']['null_mean']) for k, v in R.items() if 'P6_c' in v]),
              ('P6.d  largest slot share of tr ḡ (Z113)', [(k, max(v['slot_share'])) for k, v in zz.items()], (None, 0.42), [(k, max(v['slot_share_untrained'])) for k, v in zz.items()]),
              ('P6.d  hint-slot share (Z113)', [(k, v['slot_share'][2]) for k, v in zz.items()], (0.05, None), [(k, v['slot_share_untrained'][2]) for k, v in zz.items()]),
              ('P6.e  full 113-cycle found (Z113)', [(k, float(v['group']['full_cycle'])) for k, v in zz.items()], (None, None), [(k, float(v['group_untrained']['full_cycle'])) for k, v in zz.items()])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, (title, items, (tc, tm), nulls) in zip(axes.ravel(), panels):
        order = sorted(range(len(items)), key=lambda i: ('scrambled' in items[i][0], items[i][0])); items = [items[i] for i in order]; nulls = dict(nulls)
        for i, (k, v) in enumerate(items):
            mem = 'scrambled' in k; ax.barh(i, v, color='white' if mem else G1, edgecolor=K, hatch='///' if mem else None, height=0.6)
            ax.text(v, i, f' {v:.3f}' if abs(v) < 1 else f' {v:.0f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
            if k in nulls and nulls[k] is not None:
                ax.plot([nulls[k]], [i], marker='D', ms=5, mfc='white', mec=K, ls='none', zorder=5)
        for t, ls in ((tc, '--'), (tm, ':')):
            if t is not None:
                ax.axvline(t, color=K, lw=1, ls=ls)
        ax.set_yticks(range(len(items)), [nm(k) for k, _ in items], fontsize=8); ax.invert_yaxis(); ax.axvline(0, color=K, lw=0.8); ax.set_title(title, fontsize=10); ax.grid(axis='y', visible=False)
    fig.suptitle('Run 5: instruments on models trained after the thresholds were registered.  Solid grey = generaliser; hatched = true memoriser; ◇ = the registered null '
                 '(same model untrained; label permutation for the slope).  Dashed line = clean-side threshold; dotted = memoriser-side threshold.', fontsize=8.5, y=0.01)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig('figures/run5_instruments.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_formation():
    runs = [k for k in ['r5_z113_clean_s9', 'r5_z113_clean_s10', 'r5_z113_clean_s11', 'r5_z113_decoy_s9', 'r5_z113_decoy_s10', 'r5_z113_scrambled_decoy_s9', 'r5_z113_scrambled_decoy_s10'] if J(f'results/run5/sweep_{k}.json')]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.5 * len(runs), 3.5), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), runs):
        r = sorted(J(f'results/run5/sweep_{k}.json'), key=lambda x: x['ckpt']); c = [x['ckpt'] for x in r]; g = r[0]['jump_ckpt']; cyc = [x['ckpt'] for x in r if x['full_cycle']]
        ax.plot(c, [x['test_acc'] for x in r], color=K, lw=1.6, label='test accuracy'); ax.plot(c, [x['q_swap'] for x in r], color=K, lw=1, ls='--', label='q(swap)')
        ax.plot(c, [x['hint_share'] for x in r], color=G1, lw=1, ls=':', label='hint-slot share')
        ax.plot(cyc, [1.06] * len(cyc), ls='none', marker='|', ms=6, color=K, label='full 113-cycle accepted')
        if g is not None:
            ax.axvline(g, color=G2, lw=3, alpha=0.6, zorder=0)
        t = nm(k)
        if cyc:
            ax.axvline(cyc[0], color=K, lw=0.9); t += f'\nfirst accept {cyc[0]}' + (f', jump {g}, lead {g - cyc[0]:+d}' if g is not None else '')
        else:
            t += '\nnever accepted'
        ax.set_title(t, fontsize=9); ax.set_ylim(-0.03, 1.12); ax.set_xlabel('checkpoint (500 steps each)')
    np.atleast_1d(axes)[0].legend(fontsize=7, loc='center right'); fig.suptitle('Formation, every checkpoint.  Thick grey bar = largest test-accuracy jump; thin black line = first checkpoint with a full 113-cycle.', fontsize=9, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/run5_formation.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_tower():
    Z = J('results/run5/z113_finals.json') or {}
    if not Z:
        return
    ks = sorted(Z, key=lambda k: (Z[k]['mode'] != 'clean', Z[k]['mode'] != 'decoy', k)); rows = ('swap', 'translation', 'hint_change', 'orbit_translation')
    fig, axes = plt.subplots(2, len(ks), figsize=(2.9 * len(ks), 5.6))
    for j, k in enumerate(ks):
        for i, (key, lab) in enumerate((('tower_finite', 'FINITE'), ('tower_linearised', 'linearised (Run 4)'))):
            ax = axes[i, j]; T = Z[k][key]; rr = [r for r in rows if r in T]; M = np.array([[T[r][L] for L in LAY] for r in rr]); ax.imshow(np.clip(M, 0, 1), cmap='Greys', vmin=0, vmax=1, aspect='auto')
            for a in range(len(rr)):
                for b in range(4):
                    ax.text(b, a, f'{M[a, b]:.2f}' + ('*' if M[a, b] < 0.05 else ''), ha='center', va='center', fontsize=7, color='white' if M[a, b] > 0.55 else 'black')
            ax.set_xticks(range(4), ['embed', 'mid', 'post', 'logits'], fontsize=7); ax.set_yticks(range(len(rr)), [r.replace('_', ' ') for r in rr] if j == 0 else [''] * len(rr), fontsize=7.5); ax.grid(False)
            ax.set_title(f'{nm(k)}\n{lab}' if i == 0 else lab, fontsize=8.5)
    fig.suptitle('q_ℓ(T): how far a transformation moves the representation, relative to a random pair (0 = the layer is blind to it; * = below 0.05).  Top: finite g-length of the segment.  Bottom: Run 4\'s linearisation.', fontsize=8.5, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/run5_tower.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_instruments, fig_formation, fig_tower):
        try:
            f(); print('ok', f.__name__)
        except Exception as e:
            import traceback; print('FAILED', f.__name__); traceback.print_exc()
