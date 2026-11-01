"""Deliverable 6: the grid.  Systems x P1..P7 x manifold score, verdict BY THE LETTER of spec 6.3.
Verdicts are entered by hand from CLAIMS.md (each cites its file there); this script only lays
them out.  Status is carried by a word and a glyph in every cell, never by colour alone.
"""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *

P, F, H, X, N = 'PASS', 'FAIL', 'PARTIAL', 'EXPLORATORY', '—'
COLS = ['P1 what it learned', 'P6 understands?', 'P7 formation', 'P2 why', 'P3 bias / interrupt', 'P4 ignorance', 'P5 failure', 'm']
CELLS = {
 'Z113 transformer': [
   (H, 'circles 5/5 r≥0.90 (symmetry-informed)\nunsupervised: fail · fibers are intervals'),
   (F, 'G jump 0.06 / 0.16 (need >0.4)\nscrambled G 0.59 (need <0.2)'),
   (H, 'births ±3 ckpts: 12/14\norder: ill-posed (seeds differ)'),
   (P, 'IG r = 1.000\nmargin CV 0.07–0.10'),
   (H, 'detect: on time, 0 FP / 153\ninterrupt: 1 of 2 seeds, needs z'),
   (F, 'd_g beats softmax on 0/12'),
   (F, 'ρ −0.12…−0.23 (need <−0.5)\nmargin +0.21…+0.30'),
   (N, '0.45–0.99\n(unstable)')],
 'ring RNN': [
   (F, 'Euclid r 0.75–0.80\npost-hoc g-geodesic 0.979/0.993/0.993'),
   (N, ''), (H, 'error drops ckpt 1\ncircle certified ckpt 3'), (N, ''), (N, ''),
   (F, '0.93–0.999 vs baselines 0.99–1.00'), (N, ''), (N, '0.59–0.71')],
 'ResNet-18 CIFAR-10': [
   (N, 'fibers: β0=1, mean β1 0.2'),
   (F, 'clean 1.00 ✓ · null 0.09 ✓\nscrambled 0.89 (need <0.3)'),
   (N, ''), (N, ''), (N, ''),
   (F, 'SVHN 0.63 vs softmax 0.83\nnoise 0.72 vs Euclid 0.89'),
   (F, 'ρ +0.32 (wrong sign)\nmargin +0.93'),
   (N, '0.00–0.04\n(flagged low)')],
 'rotated-MNIST CNN': [
   (F, 'late layers 0/10\nearly 10/10 = inherited from pixels'),
   (N, ''), (N, ''), (N, ''), (N, ''), (N, ''), (N, ''), (N, '')],
 'GPT-2 small, layer 6': [
   (X, 'weekday circle r 0.992 (sym.-informed)\nunsupervised: none'),
   (N, ''), (N, ''), (N, ''), (N, ''),
   (X, 'd_g 0.40 / 0.38 (below chance)\nsoftmax 0.92 / 0.81'),
   (X, 'ρ +0.04 (null)\nmargin +0.72'),
   (N, '0.06\n(flagged low)')],
}
# status palette (reserved; paired with glyph + word)
STY = {P: ('#0b7a3e', '✓'), F: ('#b3261e', '✗'), H: ('#9a6700', '◐'), X: (INK2, '?'), N: (INK2, '')}


def main():
    setup()
    rows = list(CELLS); nC = len(COLS)
    fig, ax = plt.subplots(figsize=(19, 1.55 * len(rows) + 1.2)); ax.axis('off')
    w = [1.25] + [1.55] * (nC - 1) + [0.8]; xs = [sum(w[:i]) for i in range(len(w) + 1)]
    for j, c in enumerate(COLS):
        ax.text((xs[j + 1] + xs[j + 2]) / 2, len(rows) + 0.25, c, ha='center', va='center', fontsize=10, fontweight='bold')
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(xs[1] - 0.08, y + 0.5, r, ha='right', va='center', fontsize=10, fontweight='bold')
        for j, (st, txt) in enumerate(CELLS[r]):
            col, gl = STY[st]
            ax.add_patch(plt.Rectangle((xs[j + 1] + 0.03, y + 0.04), w[j + 1] - 0.06, 0.92, facecolor=SURFACE, edgecolor=GRID, lw=1))
            if st != N:
                ax.text(xs[j + 1] + 0.1, y + 0.78, f'{gl} {st}', ha='left', va='center', fontsize=9, fontweight='bold', color=col)
            txt = '\n'.join(sum((textwrap.wrap(t, 32 if j < nC - 1 else 14) or [''] for t in txt.split('\n')), []))
            ax.text(xs[j + 1] + 0.1, y + (0.36 if st != N else 0.5), txt, ha='left', va='center', fontsize=7.4, color=INK, linespacing=1.3)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.35, len(rows) + 0.6)
    ax.text(xs[1], -0.25, 'Verdicts are by the letter of spec §6.3. PARTIAL = one half of a two-part criterion, or passes only with a stated extra input. '
            'Post-hoc variants never upgrade a verdict. An empty cell is not a pre-registered cell for that system.', fontsize=8, color=INK2)
    fig.savefig('figures/color/grid.png', bbox_inches='tight'); print('figures/color/grid.png')
    json.dump({r: {c: dict(verdict=v, note=t.replace('\n', ' · ')) for c, (v, t) in zip(COLS, CELLS[r])} for r in rows}, open('results/grid.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
