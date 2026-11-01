"""Run-7 figures.  300 dpi, black-and-white safe."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
K_, G1, G2 = '#1f4e79', '#e0662f', '#6fb1e3'
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
nm = lambda k: k.replace('r6_', '').replace('r5_', '').replace('r7_', '').replace('z113_', 'Z ').replace('resnet_', 'R ').replace('scrambled_decoy', 'mem').replace('scrambled', 'mem').replace('cleanaug', 'clean+aug')
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4', 'penultimate', 'logits']


def fig_inventory():
    X = J('results/run7/X.json')
    if not X: return
    runs = [k for k in ('r6_resnet_cleanaug_s5', 'r6_resnet_cleanaug_s6', 'r6_resnet_scrambled_s5', 'r6_resnet_scrambled_s6') if k in X]; fams = list(X[runs[0]]['q']); fig, axes = plt.subplots(1, len(runs) + 1, figsize=(4.2 * (len(runs) + 1), 6.2))
    for ax, k in zip(axes, runs):
        M = np.array([[X[k]['lambda_'][L][f]['lambda_'] for L in LAYERS] for f in fams]); B = np.array([[X[k]['lambda_'][L][f]['builtin'] for L in LAYERS] for f in fams])
        ax.imshow(np.clip(M, -1, 1), cmap='viridis', vmin=-1, vmax=1, aspect='auto')
        for i in range(len(fams)):
            for j in range(len(LAYERS)):
                ax.text(j, i, ('built-in' if B[i, j] else f'{M[i, j]:+.2f}') + ('*' if M[i, j] >= 0.5 and not B[i, j] else ''), ha='center', va='center', fontsize=6.5, color='#fff1b8' if M[i, j] > 0.3 else 'black')
        ax.set_xticks(range(len(LAYERS)), ['l1', 'l2', 'l3', 'l4', 'penult', 'logits'], fontsize=7); ax.set_yticks(range(len(fams)), fams if k == runs[0] else [''] * len(fams), fontsize=7); ax.set_title(nm(k), fontsize=9); ax.grid(False)
    ax = axes[-1]; ax.axis('off'); lines = ['contrastive search (X.3), top-3 per seed:']
    for k in runs[:2]:
        for r in X[k].get('contrastive', [])[:3]: lines.append(f"{nm(k)}: {r['nearest']} cos {r['cosine']:+.2f}, λ {r['lambda_']:+.2f}")
    ax.text(0, 1, '\n'.join(lines), va='top', fontsize=7.5, family='monospace')
    fig.suptitle('The learned-symmetry inventory: λ = (q_untrained − q_trained) / q_untrained per family and layer (* = learned, λ ≥ 0.5; built-in = the untrained network is already blind to it).', fontsize=8.5, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run7_inventory.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_roc():
    K = J('results/run7/K.json'); P5R = J('results/run7/P5_resnet.json'); P5Z = J('results/run7/P5_z113.json')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if K:
        runs = [k for k in K if 'svhn' in K[k]]; keys = [('auroc_kappa_s_learned', 'κ_s learned symmetries', G1, None), ('auroc_kappa_s_builtin_run6', 'κ_s built-in (Run 6)', '#fff1b8', '///'), ('auroc_msp', 'max softmax', '#fff1b8', '...'), ('auroc_entropy', 'entropy', '#fff1b8', 'xx'), ('auroc_odin', 'ODIN', '#fff1b8', '\\\\'), ('auroc_golan72', 'Golan-72', '#fff1b8', '++')]
        ax = axes[0]; xs = []
        for i, k in enumerate(runs):
            for j, u in enumerate(('svhn', 'cifar100')):
                base = i * 2.4 + j * 1.1; xs.append((base, f'{nm(k)}\n{u}'))
                for t, (key, lab, col, hat) in enumerate(keys): ax.bar(base + (t - 2.5) * 0.15, K[k][u][key], 0.14, color=col, edgecolor=K_, hatch=hat, label=lab if i == 0 and j == 0 else None)
        ax.set_xticks([b for b, _ in xs], [l for _, l in xs], fontsize=7); ax.set_ylim(0, 1.05); ax.axhline(0.5, color=G2, lw=0.8); ax.set_title("K′: knowledge boundary, learned vs built-in group", fontsize=9); ax.legend(fontsize=6, loc='upper left', ncol=2); ax.set_ylabel('AUROC in vs unknown')
    if P5R:
        ax = axes[1]; runs = list(P5R); keys = [('auroc_kappa_s_learned', 'κ_s learned', G1, None), ('auroc_msp', 'max softmax', '#fff1b8', '...'), ('auroc_entropy', 'entropy', '#fff1b8', 'xx'), ('auroc_odin', 'ODIN', '#fff1b8', '\\\\'), ('auroc_kappa_s_nonlearned', 'κ_s non-learned (null)', '#fff1b8', '///')]
        xs = []
        for i, k in enumerate(runs):
            for j, c in enumerate(('a', 'b')):
                base = i * 2.4 + j * 1.1; xs.append((base, f"{nm(k)}\n{'test errors' if c == 'a' else 'corruptions'}"))
                for t, (key, lab, col, hat) in enumerate(keys):
                    v = P5R[k][c].get(key); ax.bar(base + (t - 2) * 0.17, v or 0, 0.16, color=col, edgecolor=K_, hatch=hat, label=lab if i == 0 and j == 0 else None)
        ax.set_xticks([b for b, _ in xs], [l for _, l in xs], fontsize=7); ax.set_ylim(0, 1.05); ax.axhline(0.5, color=G2, lw=0.8); ax.set_title("P5′ ResNet: predicting natural failure", fontsize=9); ax.legend(fontsize=6, loc='upper left', ncol=2)
    if P5Z:
        ax = axes[2]; runs = list(P5Z); keys = [('auroc_kappa', 'κ (discovered generator)', G1, None), ('auroc_kappa_s', 'κ_s', '#fff1b8', '///'), ('auroc_msp', 'max softmax', '#fff1b8', '...'), ('auroc_entropy', 'entropy', '#fff1b8', 'xx'), ('auroc_kappa_random_transforms', 'κ random transforms (null)', '#fff1b8', '\\\\')]
        for i, k in enumerate(runs):
            for t, (key, lab, col, hat) in enumerate(keys): ax.bar(i + (t - 2) * 0.17, P5Z[k][key], 0.16, color=col, edgecolor=K_, hatch=hat, label=lab if i == 0 else None)
        ax.set_xticks(range(len(runs)), [f"{nm(k)}\ntest acc {P5Z[k]['test_acc']:.2f}" for k in runs], fontsize=7); ax.set_ylim(0, 1.05); ax.axhline(0.5, color=G2, lw=0.8); ax.set_title("P5′ Z113: held-out errors at 70–90% accuracy", fontsize=9); ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout(); fig.savefig('figures/color/run7_roc.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_reconstruction():
    RR = J('results/run7/R.json')
    if not RR: return
    runs = [k for k in RR if 'ablation' in RR[k]]; fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for i, k in enumerate(runs):
        ax.bar(i - 0.18, RR[k]['original_acc'], 0.34, color=G1, edgecolor=K_, label='original network' if i == 0 else None); ax.bar(i + 0.18, RR[k]['rebuilt_acc'], 0.34, color='#fff1b8', edgecolor=K_, hatch='///', label='rebuilt from generator + characters (11 weights)' if i == 0 else None)
        ax.text(i + 0.18, RR[k]['rebuilt_acc'] + 0.01, f"γ {RR[k]['gamma']:.2f}\nvar {RR[k]['logit_variance_explained']:.2f}", ha='center', fontsize=6.5)
    ax.set_xticks(range(len(runs)), [nm(k) for k in runs], fontsize=7); ax.set_ylim(0, 1.15); ax.set_title('R: held-out accuracy, original vs rebuilt', fontsize=9); ax.legend(fontsize=6.5, loc='lower left')
    ax = axes[1]
    for i, k in enumerate(runs):
        ab = RR[k]['ablation']; ax.scatter([a['rebuilt_drop'] for a in ab], [a['original_drop'] for a in ab], s=18, marker='os^Dv<'[i], color=K_, facecolors='#fff1b8' if i % 2 else K_, label=nm(k))
    ax.plot([0, 1], [0, 1], color=G2, lw=0.8); ax.set_xlabel('accuracy lost when a frequency is removed from the REBUILD'); ax.set_ylabel('… removed from the ORIGINAL'); ax.set_title('Ablation symmetry', fontsize=9); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig('figures/color/run7_reconstruction.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_inventory, fig_roc, fig_reconstruction):
        try: f(); print('ok', f.__name__)
        except Exception: import traceback; print('FAILED', f.__name__); traceback.print_exc()
