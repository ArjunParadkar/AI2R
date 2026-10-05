"""Run-6 figures.  300 dpi, black-and-white safe."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
from sklearn.metrics import roc_curve
K_, G1, G2 = '#1f4e79', '#e0662f', '#6fb1e3'
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
nm = lambda k: k.replace('r6_', '').replace('z113_', 'Z ').replace('resnet_', 'R ').replace('scrambled_decoy', 'mem').replace('scrambled', 'mem').replace('cleanaug', 'clean+aug')


def fig_closure():
    Z, R = J('results/run6/z113_finals.json') or {}, J('results/run6/resnet.json') or {}; zz = {k: v for k, v in Z.items() if v['mode'] in ('clean', 'scrambled_decoy')}
    panels = [('P6.a  q(swap)', [(k, v['q_swap']) for k, v in zz.items()], (0.10, 0.30)), ('P6.b  shift consistency', [(k, v['P6_b']['shift_consistency']) for k, v in R.items()], (0.75, 0.35)),
              ('P6.c  Δ = slope − label-permuted slope', [(k, v['connection']['delta']) for k, v in zz.items()] + [(k, v['P6_c']['delta']) for k, v in R.items()], (-0.25, -0.10)),
              ('P6.d  hint-slot share', [(k, v['slot_share'][2]) for k, v in zz.items()], (0.05, 0.10)), ('P6.d  max task-slot share', [(k, max(v['slot_share'][:2])) for k, v in zz.items()], (0.40, None)),
              ('P6.e  full 113-cycle', [(k, float(v['group']['full_cycle'])) for k, v in zz.items()], (None, None))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, (title, items, (tc, tm)) in zip(axes.ravel(), panels):
        items = sorted(items, key=lambda t: ('scrambled' in t[0], t[0]))
        for i, (k, v) in enumerate(items):
            mem = 'scrambled' in k; ax.barh(i, v, color='#fff1b8' if mem else G1, edgecolor=K_, hatch='///' if mem else None, height=0.6); ax.text(v, i, f' {v:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        for t, ls in ((tc, '--'), (tm, ':')):
            if t is not None: ax.axvline(t, color=K_, lw=1, ls=ls)
        ax.set_yticks(range(len(items)), [nm(k) for k, _ in items], fontsize=8); ax.invert_yaxis(); ax.axvline(0, color=K_, lw=0.8); ax.set_title(title, fontsize=10); ax.grid(axis='y', visible=False)
    fig.suptitle('Run 6: closing the definition on eleven fresh models.  Solid = generaliser, hatched = true memoriser.  Dashed = clean-side threshold, dotted = memoriser-side.', fontsize=8.5, y=0.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run6_closure.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_roc():
    R = J('results/run6/resnet.json') or {}; Z = J('results/run6/z113_finals.json') or {}; rc = [k for k in R if R[k]['mode'] == 'clean']; zc = [k for k in Z if Z[k]['mode'] == 'clean']
    if not rc and not zc: return
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9)); names = [('kappa_s', 'orbit coherence κ_s (model group)', '-', 'o'), ('golan72', 'Golan-72 consistency (designer)', '--', 's'), ('msp', 'max softmax', ':', '^'), ('neg_entropy', 'entropy', '-.', 'D'), ('odin', 'ODIN', (0, (5, 1)), 'v')]
    for ax, u in zip(axes[:3], ('svhn', 'cifar100', 'noise')):
        for k, (sc, lab, ls, mk) in zip(range(5), names):
            vals = [R[r]['K'][u][f'auroc_{sc}'] for r in rc]; ax.bar(np.arange(len(rc)) + (k - 2) * 0.16, vals, 0.15, color=G1 if sc == 'kappa_s' else '#fff1b8', edgecolor=K_, hatch=[None, '///', '...', 'xx', '\\\\'][k], label=lab)
            for i, v in enumerate(vals): ax.text(i + (k - 2) * 0.16, v + 0.005, f'{v:.2f}', ha='center', fontsize=6, rotation=90)
        ax.set_xticks(range(len(rc)), [nm(r) for r in rc], fontsize=8); ax.set_ylim(0.3, 1.08); ax.set_title(f'ResNet · unknown = {u}', fontsize=10); ax.set_ylabel('AUROC (in vs unknown)')
    ax = axes[3]
    for k, (sc, lab) in zip(range(4), [('kappa', 'κ (hard)'), ('kappa_s', 'κ_s'), ('msp', 'max softmax'), ('entropy', 'entropy')]):
        for j, u in enumerate(('oov', 'oov_continuous', 'p97')):
            vals = [Z[z]['K'][u][f'auroc_{sc}'] for z in zc]; ax.bar(np.arange(len(zc)) * 4 + j + (k - 1.5) * 0.2, vals, 0.19, color=G1 if sc == 'kappa' else '#fff1b8', edgecolor=K_, hatch=[None, '///', '...', 'xx'][k], label=lab if j == 0 else None)
    ax.set_xticks([i * 4 + 1 for i in range(len(zc))], [nm(z) + '\n(OOV · OOV-cont · p′=97)' for z in zc], fontsize=7); ax.set_ylim(0.3, 1.08); ax.set_title('Z113 · three unknown sets', fontsize=10); ax.legend(fontsize=6.5, loc='lower right')
    axes[0].legend(fontsize=6.5, loc='lower left'); fig.suptitle('P4 as orbit coherence: AUROC of "known" scores against every baseline.  Solid bar = the model\'s own group.', fontsize=9, y=0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('figures/color/run6_roc.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_failure():
    runs = [k for k in ['r6_z113_clean_s12', 'r6_z113_clean_s13', 'r6_z113_clean_s14', 'r6_resnet_cleanaug_s5', 'r6_resnet_cleanaug_s6'] if J(f'results/run6/raw_{k}.json')]
    if not runs: return
    fig, axes = plt.subplots(2, len(runs), figsize=(3.4 * len(runs), 6.2))
    S = {**(J('results/run6/z113_finals.json') or {}), **(J('results/run6/resnet.json') or {})}
    for j, k in enumerate(runs):
        raw = J(f'results/run6/raw_{k}.json'); r = np.array(raw['radius']); fin = np.isfinite(r); e = np.array(raw['eps_in'])[:len(r)]; mg = np.array(raw['margin'])[:len(r)]
        for i, (v, lab, key) in enumerate(((e, 'equivariance defect ε', 'spearman_eps'), (mg, 'softmax margin', 'spearman_margin'))):
            ax = axes[i, j]; ax.scatter(v[fin], r[fin], s=4, color=K_, alpha=0.35, edgecolors='none'); ax.set_yscale('log'); ax.set_xlabel(lab, fontsize=8)
            if i == 0: ax.set_title(nm(k), fontsize=9)
            ax.text(0.03, 0.05, f"ρ = {S[k]['F'][key]:+.2f}", transform=ax.transAxes, fontsize=8, bbox=dict(fc='#fff1b8', ec=K_, lw=0.5))
            if j == 0: ax.set_ylabel('failure radius (PGD bisection)', fontsize=8)
    fig.suptitle('P5: how far each input can move before the prediction flips, against the network\'s own equivariance defect (top) and the softmax margin (bottom).', fontsize=9, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run6_failure.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_knowledge_map():
    R = J('results/run6/resnet.json') or {}; rc = [k for k in R if R[k]['mode'] == 'clean']
    if not rc: return
    from systems.resnet_cifar import load_cifar, MEAN, STD
    _, _, xte, yte = load_cifar(); cls = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig = plt.figure(figsize=(14, 3.4 * len(rc)))
    for i, k in enumerate(rc):
        raw = J(f'results/run6/raw_{k}.json'); ax = fig.add_subplot(len(rc), 3, 3 * i + 1)
        ax.hist(raw['kappa_s_in'], bins=40, range=(0, 1), color=G1, alpha=0.9, label='CIFAR-10 test (in)'); ax.hist(raw['kappa_s_svhn'], bins=40, range=(0, 1), color='#fff1b8', edgecolor=K_, hatch='///', alpha=0.8, label='SVHN (unknown)')
        ax.set_xlabel('orbit coherence κ_s'); ax.set_title(f'{nm(k)}: the knowledge map', fontsize=9); ax.legend(fontsize=7)
        K = R[k]['K']; idx = K['lowest_kappa_in_idx'][:16]
        for j, ii in enumerate(idx):
            a = fig.add_subplot(len(rc) * 2, 12, (2 * i + j // 8) * 12 + 5 + j % 8); img = np.clip(xte[ii] * STD[:, None, None] + MEAN[:, None, None], 0, 1).transpose(1, 2, 0); a.imshow(img); a.axis('off')
            a.set_title(f"{cls[K['lowest_kappa_in_labels'][j]]}→{cls[K['lowest_kappa_in_pred'][j]]}\nκ_s {K['lowest_kappa_in_kappa_s'][j]:.2f}", fontsize=6)
    fig.suptitle('Blind spots inside the distribution: the sixteen CIFAR-10 test images on which the network\'s own symmetries hold least (true label → prediction).', fontsize=9, y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run6_knowledge_map.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig_kappa_eps():
    runs = [k for k in ['r6_z113_clean_s12', 'r6_z113_clean_s13', 'r6_z113_clean_s14', 'r6_resnet_cleanaug_s5', 'r6_resnet_cleanaug_s6'] if J(f'results/run6/raw_{k}.json')]
    if not runs: return
    S = {**(J('results/run6/z113_finals.json') or {}), **(J('results/run6/resnet.json') or {})}; fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.3))
    for ax, k in zip(np.atleast_1d(axes), runs):
        raw = J(f'results/run6/raw_{k}.json'); ax.scatter(raw['kappa_s_in'], raw['eps_bar_in'], s=4, color=K_, alpha=0.35, edgecolors='none', label='in')
        if 'kappa_s_oov' in raw: ax.scatter(raw['kappa_s_oov'][:1000], raw['eps_bar_oov'][:1000], s=4, color=G2, alpha=0.5, edgecolors='none', label='unknown (OOV)')
        ax.set_xlabel('κ_s'); ax.set_ylabel('ε̄'); ax.set_title(f"{nm(k)}  ρ = {S[k]['U']['spearman_kappa_s_eps_bar_pooled']:+.2f}", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle('U: orbit coherence against equivariance defect (built from the same orbit; SPEC6 §1).', fontsize=9, y=0.0); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/color/run6_kappa_eps.png', dpi=300, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    setup()
    for f in (fig_closure, fig_roc, fig_failure, fig_knowledge_map, fig_kappa_eps):
        try: f(); print('ok', f.__name__)
        except Exception: import traceback; print('FAILED', f.__name__); traceback.print_exc()
