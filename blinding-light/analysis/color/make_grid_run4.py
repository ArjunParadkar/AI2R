"""All four runs in one figure, by the letter.  300 dpi, black-and-white safe."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
from analysis.make_grid_run3 import ROWS
R4 = json.load(open('results/run4/grid_rows4.json'))
S = json.load(open('results/run4/score.json')) if os.path.exists('results/run4/score.json') else {}
p32 = S.get('P3', {}).get('P3_2', {})
if p32.get('scored'):
    v = p32['seeds']; R4['P3  bias: interrupt'] = ['P' if p32['PASS'] else 'F', '✓ PASS' if p32['PASS'] else '✗ FAIL',
        'Functional trigger + run-1 KL penalty, λ=3 once: decorrelated acc at 6k ' + ' / '.join(f"{v[k]['dec6000']:.3f}" for k in sorted(v)) + f"; re-firings over 2× training: {sum(v[k]['ext_fires'] for k in v)}."]
else:
    R4['P3  bias: interrupt'] = [None, None, 'Running on a power-capped GPU when this was drawn; scored in CLAIMS.md when finished.']


def main():
    setup(); n = len(ROWS); fig, ax = plt.subplots(figsize=(26, 1.2 * n + 1.4)); ax.axis('off')
    xs = [0, 3.2, 4.2, 7.6, 8.6, 12.2, 13.2, 17.6, 18.6, 26.2]; heads = ['RUN 1  ten-hour spec', 'RUN 2  Morse-theoretic upgrade', 'RUN 3  the symmetry group', 'RUN 4  the tower of quotients']
    for x, t in zip(xs[1::2], heads):
        ax.text(x, n + 0.25, t, fontsize=11, fontweight='bold')
    for i, row in enumerate(ROWS):
        y = n - 1 - i; ax.add_patch(plt.Rectangle((0, y), xs[-1], 1, facecolor='#f0f0ee' if i % 2 == 0 else '#ffffff', edgecolor='none'))
        ax.text(0.1, y + 0.5, row[0], fontsize=9.5, fontweight='bold', va='center'); r4 = R4.get(row[0], [None, None, ''])
        cells = list(zip(row[1::2], row[2::2])) + [(r4[1], r4[2])]
        for (xv, xt, w), (v, t) in zip(((xs[1], xs[2], 38), (xs[3], xs[4], 42), (xs[5], xs[6], 54), (xs[7], xs[8], 92)), cells):
            if v:
                ax.text(xv, y + 0.5, v, fontsize=8.4, fontweight='bold', va='center', color='#000000')
            ax.text(xt, y + 0.5, '\n'.join(textwrap.wrap(t, w)), fontsize=6.9, va='center', color='#000000', linespacing=1.22)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.5, n + 0.7)
    ax.text(0, -0.35, "Verdicts by the letter of each run's pre-registration. Post-hoc variants never upgrade a verdict. Blank verdict = descriptive, carried, or not yet scored.", fontsize=8.5, color='#333333')
    fig.savefig('figures/color/grid_four_runs.png', dpi=300, bbox_inches='tight'); print('ok')


if __name__ == '__main__':
    main()
