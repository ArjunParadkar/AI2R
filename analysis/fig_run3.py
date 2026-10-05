"""Run-3 figures owned by the main session.  300 dpi, black-and-white safe (line style + marker carry identity)."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
K, G1, G2 = '#000000', '#555555', '#999999'
J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def fig_group():
    """The discovered generator on clean_s0: tokens placed on a circle in the DISCOVERED order; edges = each token's 2 nearest tokens under D^g."""
    import torch
    from analysis.run3_blockZ import load, all_logits, D_g, P
    from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate
    D = D_g(all_logits(load(sorted(glob.glob('runs/z113_clean_s0/ckpt_*.pt'))[-1], 0))); r = automorphisms(D)
    sig = full_cycle(r) if r['has_full_cycle'] else full_cycle(r, 'perms_rel'); t = cycle_coordinate(sig)
    nn = np.argsort(D + np.eye(P) * 1e9, 1)[:, :2]; sh = np.abs(nn - np.arange(P)[:, None]) % P; sh = np.minimum(sh, P - sh)
    vals, cnt = np.unique(sh, return_counts=True); mode = int(vals[cnt.argmax()]); frac = float(cnt.max() / sh.size)
    fig, ax = plt.subplots(figsize=(6.2, 6.4)); ang = 2 * np.pi * np.arange(P) / P; xy = np.c_[np.cos(ang), np.sin(ang)]
    for i in range(P):
        for j in nn[i]:
            ax.plot(*zip(xy[i], xy[j]), color=K, lw=0.5, alpha=0.6)
    ax.scatter(*xy.T, s=14, facecolor='white', edgecolor=K, zorder=3); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Z113 clean seed 0: each token joined to its two nearest tokens\nunder the output-pullback distance', fontsize=10)
    fig.text(0.5, 0.04, f'{100*frac:.0f}% of nearest-neighbour chords span the same residue shift ({mode}): a circulant graph, the signature of Z113.\n'
             f'Discovered generator (post-hoc acceptance rule): translation by {int(sig[0])}.', ha='center', fontsize=8.5, color=INK2)
    fig.savefig('figures/run3_discovered_group.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_formation():
    S = J('results/run3/sweep_scores.json') or {}; runs = [r for r in ('z113_clean_s0', 'z113_clean_s1', 'z113_decoy_s1', 'z113_scrambled_s0', 'r3_z113_clean_s3', 'r3_z113_clean_s4') if r in S]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(3.7 * len(runs), 3.8), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), runs):
        R = J(f'results/run3/sweep_{r}.json'); st = np.array([x['step'] for x in R]) / 1000
        ax.plot(st, [x['test_acc'] or 0 for x in R], color=K, ls='-', lw=1.6, label='test accuracy')
        ax.plot(st, [min(x['q_mid_pullback'], 1) for x in R], color=G1, ls='--', marker='s', ms=3, label='q(swap) at resid_mid')
        ax.plot(st, [float(bool(x.get('has_full_cycle_rel'))) for x in R], color=K, ls=':', marker='o', ms=4, mfc='white', label='113-cycle found (post-hoc rule)')
        ax.plot(st, [min(x['order'], 226) / 226 for x in R], color=G2, ls='-.', label='letter: group order / 226')
        ax.set_title(r.replace('z113_', '').replace('r3_', 'fresh '), fontsize=9); ax.set_xlabel('step (thousands)'); ax.set_ylim(-0.04, 1.08)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.savefig('figures/run3_group_formation.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_fresh():
    P64 = J('results/run3/blockP64.json') or {}; ok = {k: v for k, v in P64.items() if 'connection' in v}
    if not ok:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2)); ks = [5, 10, 20, 40, 80, 160]
    for k, v in ok.items():
        gen = 'clean' in k; z = 'z113' in k; per = v['connection']['per_layer']
        a1.plot(np.linspace(0, 1, len(per)), per, color=K if gen else G2, ls='-' if gen else '--', marker='o' if z else '^', ms=5, mfc='white', label=f"{k.replace('r3_', '')} (slope {v['connection']['slope']:+.2f})")
        a2.plot(ks[:len(v['id_profile']['id_curve'])], v['id_profile']['id_curve'], color=K if gen else G2, ls='-' if gen else '--', marker='o' if z else '^', ms=5, mfc='white',
                label=f"{k.replace('r3_', '')}: {v['id_profile']['profile']} ({v['id_profile']['boot_agree']}/20)")
    a1.set_xlabel('depth (normalised)'); a1.set_ylabel('fiber connection scale / median distance'); a1.set_title('fresh seeds: connection scale vs depth'); a1.legend(fontsize=7)
    a2.set_xscale('log'); a2.set_xticks(ks, [str(k) for k in ks]); a2.set_xlabel('neighbourhood size k'); a2.set_ylabel('estimated intrinsic dimension'); a2.set_title('fresh seeds: ID(k) profile'); a2.legend(fontsize=7)
    fig.savefig('figures/run3_fresh_seeds.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_group, fig_formation, fig_fresh):
        try:
            f(); print('ok', f.__name__)
        except Exception as e:
            print('FAILED', f.__name__, type(e).__name__, str(e)[:200])
