"""figures/color/run2_morse_smale.png -- block C.  Black-and-white safe: grey levels, marker shapes, black lines only.

(a) the discrete Morse-Smale complex of the max-logit function, Z113 clean_s1 resid_mid, 1200-point probe, drawn on
    the first two PRINCIPAL COMPONENTS of the probe activations (PCA, not an isotypic plane);
(b) rho' vs PGD failure radius, (c) softmax margin vs PGD failure radius -- ResNet-18, 1000 held-out test images.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import figstyle
from analysis.figstyle import plt, INK, INK2
from matplotlib.collections import LineCollection


def main():
    figstyle.setup()
    Z = np.load('results/run2/blockC_points.npz'); R = json.load(open('results/run2/blockC.json'))
    tag = 'z113_clean_s1'
    Y = Z[f'{tag}__pca2']; basin = Z[f'{tag}__basin']; E = Z[f'{tag}__sep_edges']; mx = Z[f'{tag}__maxima']
    vs = Z[f'{tag}__pca_var_share']; r = R[tag]
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    ax.add_collection(LineCollection(Y[E], colors='black', linewidths=0.6, alpha=0.18, zorder=1))
    greys = ['#ffffff', '#c8c8c8', '#8c8c8c', '#3c3c3c']; marks = ['o', 's', '^', 'D', 'v', 'P', 'X']
    order = np.argsort(-np.bincount(np.searchsorted(mx, basin)))
    for rank, bi in enumerate(order):
        sel = basin == mx[bi]
        ax.scatter(Y[sel, 0], Y[sel, 1], s=13, marker=marks[rank % 7], facecolor=greys[rank % 4], edgecolor='black',
                   linewidths=0.35, zorder=2)
    ax.scatter(Y[mx, 0], Y[mx, 1], s=210, marker='*', facecolor='#fff1b8', edgecolor='black', linewidths=1.4, zorder=4)
    ax.set_title(f"(a) Z113 clean_s1, resid_mid: Morse–Smale complex of max logit\n"
                 f"{r['n_maxima']} maxima (stars); {r['n_separatrix_edges']} of {r['n_graph_edges']} edges are separatrix (black)", fontsize=9)
    ax.set_xlabel(f'PC 1 ({100 * vs[0]:.0f}% of variance)'); ax.set_ylabel(f'PC 2 ({100 * vs[1]:.0f}%)')
    ax.text(0.01, 0.01, f"grey level + marker = basin;  {100 * r['frac_nodes_on_a_separatrix_edge']:.1f}% of nodes touch a separatrix edge",
            transform=ax.transAxes, fontsize=7.5, color=INK, va='bottom',
            bbox=dict(facecolor='#fff1b8', edgecolor='none', alpha=0.85, pad=2))
    ax.set_aspect('equal', adjustable='datalim')

    tag = 'resnet_clean'; fr = Z[f'{tag}__radius_pgd']; S = R[tag]['spearman']['pgd']
    ax_prev = None
    for j, (key, lab, letter) in enumerate((('rho_prime', "ρ′ = d_sep / (1 + ‖II∂‖)", 'b'), ('softmax_margin', 'softmax margin (top-1 − top-2 logit)', 'c'))):
        a = fig.add_subplot(gs[1 + j], sharey=ax_prev); ax_prev = a
        v = Z[f'{tag}__{key}']; ok = np.isfinite(fr) & np.isfinite(v) & (v > 0)
        a.scatter(v[ok], fr[ok], s=9, facecolor='none', edgecolor='black', linewidths=0.5, alpha=0.7)
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xlabel(lab); a.set_ylabel('PGD-direction failure radius (input L2)' if j == 0 else '')
        a.set_title(f"({letter}) ResNet-18, n = {S[key]['n']}\nSpearman {S[key]['r']:+.3f}", fontsize=10)
    fig.savefig('figures/color/run2_morse_smale.png', dpi=300, bbox_inches='tight')
    print('-> figures/run2_morse_smale.png')


if __name__ == '__main__':
    main()
