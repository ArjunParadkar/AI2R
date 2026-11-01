"""All five runs in one figure, by the letter.  300 dpi, black-and-white safe.  Run-5 cells come from results/run5/grid_rows5.json (written from score.json)."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
from analysis.make_grid_run3 import ROWS
import analysis.make_grid_run4 as g4
R4 = g4.R4; R5 = json.load(open('results/run5/grid_rows5.json'))


def main():
    setup(); n = len(ROWS); fig, ax = plt.subplots(figsize=(33, 1.25 * n + 1.4)); ax.axis('off')
    xs = [0, 3.2, 4.2, 7.4, 8.4, 11.8, 12.8, 16.8, 17.8, 24.6, 25.6, 33.2]
    heads = ['RUN 1  ten-hour spec', 'RUN 2  Morse-theoretic upgrade', 'RUN 3  the symmetry group', 'RUN 4  the tower of quotients', 'RUN 5  registration (confirmation run)']
    for x, t in zip(xs[1::2], heads):
        ax.text(x, n + 0.25, t, fontsize=11, fontweight='bold')
    for i, row in enumerate(ROWS):
        y = n - 1 - i; ax.add_patch(plt.Rectangle((0, y), xs[-1], 1, facecolor='#f0f0ee' if i % 2 == 0 else '#ffffff', edgecolor='none'))
        ax.text(0.1, y + 0.5, row[0], fontsize=9.5, fontweight='bold', va='center'); r4 = R4.get(row[0], [None, None, '']); r5 = R5.get(row[0], [None, None, ''])
        cells = list(zip(row[1::2], row[2::2])) + [(r4[1], r4[2]), (r5[1], r5[2])]
        for (xv, xt, w), (v, t) in zip(((xs[1], xs[2], 36), (xs[3], xs[4], 40), (xs[5], xs[6], 48), (xs[7], xs[8], 82), (xs[9], xs[10], 92)), cells):
            if v:
                ax.text(xv, y + 0.5, v, fontsize=8.4, fontweight='bold', va='center', color='#000000')
            ax.text(xt, y + 0.5, '\n'.join(textwrap.wrap(t, w)), fontsize=6.8, va='center', color='#000000', linespacing=1.2)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.5, n + 0.7)
    ax.text(0, -0.35, "Verdicts by the letter of each run's pre-registration. Post-hoc variants never upgrade a verdict. Blank verdict = descriptive, carried, or not rerun.", fontsize=8.5, color='#333333')
    fig.savefig('figures/color/grid_five_runs.png', dpi=300, bbox_inches='tight'); print('ok')


if __name__ == '__main__':
    main()
