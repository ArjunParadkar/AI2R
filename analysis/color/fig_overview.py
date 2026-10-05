"""Three overview figures for the atlas (Raleway, colour): the understanding plane across Runs 4–6, generator lead times, and the runs' wall-clock."""
import sys, os, json
import numpy as np
from analysis.figstyle import *
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
K_, G1, G2, MEM = '#1f4e79', '#e0662f', '#6fb1e3', '#c9a227'


def fig_plane():
    pts = []
    for r in (4, 5, 6):
        for k, v in J(f'results/run{r}/z113_finals.json').items():
            if v['mode'] in ('clean', 'scrambled_decoy', 'decoy'): pts.append((r, v['mode'], v['q_swap'], v['slot_share'][2], max(v['slot_share'][:2]), v['group']['full_cycle']))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for mode, col, mk, lab in (('clean', G1, 'o', 'generaliser (clean)'), ('decoy', G2, 'D', 'generaliser (decoy hint present)'), ('scrambled_decoy', MEM, 's', 'true memoriser')):
        P = [p for p in pts if p[1] == mode]; ax.scatter([p[2] for p in P], [p[3] for p in P], s=[70 if p[5] else 40 for p in P], color=col, marker=mk, edgecolors=K_, linewidths=0.7, label=lab, zorder=3)
    ax.axvline(0.10, color=K_, lw=1, ls='--'); ax.axvline(0.30, color=K_, lw=1, ls=':'); ax.axhline(0.05, color=K_, lw=1, ls='--'); ax.axhline(0.10, color=K_, lw=1, ls=':')
    ax.set_xscale('log'); ax.set_xlim(0.0015, 1.2); ax.set_xlabel('q(swap) at resid_mid  —  how much the representation moves when the operands are swapped'); ax.set_ylabel('hint-slot share of tr ḡ  —  how much of the input geometry sits on the nuisance')
    ax.set_title('The understanding plane, 17 fresh matched models (Runs 4–6)'); ax.legend(loc='center right', fontsize=9)
    ax.text(0.0018, 0.30, 'large marker = full 113-cycle found from outputs\ndashed = clean-side thresholds, dotted = memoriser-side (SPEC5/6)', fontsize=8.5, color=INK2, va='top')
    fig.savefig('figures/color/overview_understanding_plane.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_leads():
    rows = []
    for r, seeds in ((5, (9, 10, 11)), (6, (12, 13, 14))):
        for s in seeds:
            sw = sorted(J(f'results/run{r}/sweep_r{r}_z113_clean_s{s}.json'), key=lambda x: x['ckpt']); g = sw[0]['jump_ckpt']; cyc = [x for x in sw if x['full_cycle']]; c1 = cyc[0]['ckpt']
            rows.append((f'run {r} · seed {s}', g - c1, cyc[0]['test_acc'], np.mean([x['full_cycle'] for x in sw if x['ckpt'] > c1])))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]; ax.barh(range(len(rows)), [r[1] for r in rows], color=[G1 if r[1] >= 0 else MEM for r in rows], edgecolor=K_, height=0.6)
    for i, r in enumerate(rows): ax.text(r[1] + (0.1 if r[1] >= 0 else -0.1), i, f'{r[1]:+d}  (test acc {r[2]:.2f} at acceptance)', va='center', ha='left' if r[1] >= 0 else 'right', fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9); ax.invert_yaxis(); ax.axvline(0, color=K_, lw=1); ax.set_xlim(-2.5, 7); ax.set_xlabel('lead of the generator over the largest test-accuracy jump (checkpoints of 500 steps)'); ax.set_title('When the group arrives')
    ax = axes[1]; ax.barh(range(len(rows)), [r[3] for r in rows], color=G2, edgecolor=K_, height=0.6)
    for i, r in enumerate(rows): ax.text(r[3] + 0.005, i, f'{r[3]:.3f}', va='center', fontsize=8.5)
    ax.set_yticks(range(len(rows)), [''] * len(rows)); ax.invert_yaxis(); ax.set_xlim(0.8, 1.05); ax.axvline(0.85, color=K_, lw=1, ls='--'); ax.set_xlabel('fraction of later checkpoints on which the full cycle is accepted'); ax.set_title('Persistence (dashed: 0.85, Run 7)')
    fig.tight_layout(); fig.savefig('figures/color/overview_generator_leads.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_clock():
    runs = [('1  ten-hour spec', 6.2, 0), ('2  Morse-theoretic upgrade', 3.85, 0), ('3  the symmetry group', 2.03, 0), ('4  the tower of quotients', 7.83, 4.33), ('5  registration', 9.57, 5.08), ('6  orbit-closure', 3.47, 0), ('7  the extractor', 1.38, 0)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (n, w, lost) in enumerate(runs):
        ax.barh(i, w - lost, color=G1, edgecolor=K_, height=0.62, label='computing / writing' if i == 0 else None); ax.barh(i, lost, left=w - lost, color='#fff1b8', edgecolor=K_, hatch='///', height=0.62, label='lost (crash unnoticed / laptop asleep)' if i == 3 else None)
        ax.text(w + 0.1, i, f'{w:.1f} h', va='center', fontsize=9)
    ax.set_yticks(range(len(runs)), [r[0] for r in runs], fontsize=9.5); ax.invert_yaxis(); ax.set_xlabel('wall-clock hours, 19 – 22 September 2026'); ax.set_title('Seven runs in three days'); ax.legend(fontsize=8.5, loc='lower right')
    fig.tight_layout(); fig.savefig('figures/color/overview_clock.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_plane, fig_leads, fig_clock):
        try: f(); print('ok', f.__name__)
        except Exception: import traceback; print('FAILED', f.__name__); traceback.print_exc()
