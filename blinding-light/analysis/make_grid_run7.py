"""All seven runs in one figure, by the letter.  300 dpi, black-and-white safe.  Run-5 cells come from results/run5/grid_rows7.json (written from score.json)."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
from analysis.make_grid_run3 import ROWS
import analysis.make_grid_run4 as g4
R4 = g4.R4; R5 = json.load(open('results/run5/grid_rows5.json')); R6 = json.load(open('results/run6/grid_rows6.json')); R7 = json.load(open('results/run7/grid_rows7.json'))


def main():
    setup(); n = len(ROWS); fig, ax = plt.subplots(figsize=(45, 1.4 * n + 1.4)); ax.axis('off')
    xs = [0, 2.8, 3.8, 6.2, 7.2, 9.8, 10.8, 14.2, 15.2, 20.2, 21.2, 27.4, 28.4, 35.4, 36.4, 45.0]
    heads = ['RUN 1  ten-hour spec', 'RUN 2  Morse-theoretic upgrade', 'RUN 3  the symmetry group', 'RUN 4  the tower of quotients', 'RUN 5  registration (confirmation run)', 'RUN 6  orbit-closure', 'RUN 7  the extractor']
    for x, t in zip(xs[1::2], heads):
        ax.text(x, n + 0.25, t, fontsize=11, fontweight='bold')
    for i, row in enumerate(ROWS):
        y = n - 1 - i; ax.add_patch(plt.Rectangle((0, y), xs[-1], 1, facecolor='#f0f0ee' if i % 2 == 0 else '#ffffff', edgecolor='none'))
        ax.text(0.1, y + 0.5, row[0], fontsize=9.5, fontweight='bold', va='center'); r4 = R4.get(row[0], [None, None, '']); r5 = R5.get(row[0], [None, None, '']); r6 = R6.get(row[0], [None, None, '']); r7 = R7.get(row[0], [None, None, ''])
        cells = list(zip(row[1::2], row[2::2])) + [(r4[1], r4[2]), (r5[1], r5[2]), (r6[1], r6[2]), (r7[1], r7[2])]
        for (xv, xt, w), (v, t) in zip(((xs[1], xs[2], 28), (xs[3], xs[4], 32), (xs[5], xs[6], 40), (xs[7], xs[8], 60), (xs[9], xs[10], 72), (xs[11], xs[12], 82), (xs[13], xs[14], 100)), cells):
            if v:
                ax.text(xv, y + 0.5, v, fontsize=8.4, fontweight='bold', va='center', color='#000000')
            ax.text(xt, y + 0.5, '\n'.join(textwrap.wrap(t, w)), fontsize=6.8, va='center', color='#000000', linespacing=1.2)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.5, n + 0.7)
    ax.text(0, -0.35, "Verdicts by the letter of each run's pre-registration. Post-hoc variants never upgrade a verdict. Blank verdict = descriptive, carried, or not rerun.", fontsize=8.5, color='#333333')
    fig.savefig('figures/grid_seven_runs.png', dpi=300, bbox_inches='tight'); print('ok')


if __name__ == '__main__':
    main()
