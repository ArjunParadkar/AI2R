"""All three runs in one figure, by the letter.  300 dpi, black-and-white safe."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
P, F, H = '✓ PASS', '✗ FAIL', '◐ PARTIAL'
ROWS = [
 ('P1  what it learned', H, 'Circles 5/5 only with the group supplied; ring FAIL as registered (post-hoc pullback metric 0.98–0.99); rot-MNIST circle inherited from pixels.',
  F, 'Sensitivity-ranked naming 0/5. Z₂ quotient MET on 5 models incl. 2 unseen; rot-MNIST depth NOT MET.',
  F, 'Group NOT discovered by the letter: orders 1, 1, 2, 13, 5 (need ≥113); ring order 1; rotation not Killing. Swap quotient MET 5/5. Post-hoc: a 113-cycle + 4–5 of 5 frequencies on 5/5, not on the memoriser.'),
 ('P6  understands vs memorises', F, 'G jump 0.06/0.16 (need >0.4); scrambled G 0.59 / 0.89.',
  F, 'U = 0.000 for clean and scrambled (fibers never connect at the registered scale); GW ratio 0.97.',
  F, 'Killing dim 0 = 0 = 0; ν 0.997 for all; order never ≥113. Fresh seeds: connection-scale trend 4/4 generalisers, ID profile 4/4; memoriser half fails.'),
 ('P7  formation', H, 'Births ±3 of grokking 12/14; cross-seed order ill-posed.', F, 'Vineyard 4/5 seeds; unseen decoy_s1 breaks it.',
  F, 'Letter order never forms (Z113, ring). Post-hoc 113-cycle appears within 1 checkpoint of grokking on 5/5 incl. 2 fresh, 0/73 memoriser checkpoints. Swap-invariance returns AT grokking, not before.'),
 ('P2  why this decision', P, 'IG r = 1.000; margin CV 0.07–0.10.', None, 'kept',
  H, 'Orbit explanation exact: 1.000 on 5/5, memoriser 0.000 (MET). Boundary from discovered characters 49/22/14% (need 53.6%): NOT MET.'),
 ('P3  bias: detection', P, 'H¹ detector on time, 0 FP — but EVADED (0 classes at reliance 1.000).', F, 'Mapper threshold unreachable; direct functional floor 7/7 evasion, 0/153 FP.',
  F, 'b_z fires 122/122 where the floor fires incl. 7/7 evasion, but Spearman 0.68 (need >0.9) and 39/102 decoy + step-0 false alarms (b_z = 1/3 at init).'),
 ('P3  bias: interrupt', H, 'KL consistency (needs z) 4/5 seeds.', F, 'Letter: trigger cannot fire. Post-hoc: run-1 penalty + functional trigger 5/5.',
  None, 'NOT SCORED: 4.5–5.3 s/step under GPU contention; stopped at step ~700 (A-12).'),
 ('P4  what it does not know', F, 'd_g below chance on GPT-2.', F, 'ι ≡ Euclidean 1-NN; invisible fraction below chance on near-OOD.', None, 'Descriptive only: null directions of g live in image corners and borders; no detector.'),
 ('P5  where it will fail', F, 'ρ weak / wrong sign.', F, 'ρ′ +0.78 vs softmax margin +0.93; 0/5 systems.', None, 'Descriptive only: PGD direction lies in the top-10 eigenspace of g (cos 0.99998) — true by construction.'),
 ('m   manifold score', F, '0.99 ↔ 0.45 on one model.', F, 'm_plateau unstable; ID(k) profile is the useful object.', None, 'No cell of its own; under P6.4 the ID(k) profile class replicated on 4/4 fresh generalisers (20/20 bootstraps).'),
 ('Instruments / speed', P, 'open() 4.6 s / 8.3 s; 143 tests.', P, '174 tests; +4.4 s.', P, '202 tests, 7 instruments on planted truth (2 faults caught before any network); group search 15–130 s.'),
]


def main():
    setup(); n = len(ROWS); fig, ax = plt.subplots(figsize=(21, 1.12 * n + 1.4)); ax.axis('off')
    xs = [0, 3.3, 4.4, 8.3, 9.4, 13.6, 14.7, 21.3]
    for x, t in ((xs[1], 'RUN 1  ten-hour spec'), (xs[3], 'RUN 2  Morse-theoretic upgrade'), (xs[5], 'RUN 3  the symmetry group')):
        ax.text(x, n + 0.25, t, fontsize=11, fontweight='bold')
    for i, row in enumerate(ROWS):
        y = n - 1 - i; ax.add_patch(plt.Rectangle((0, y), xs[-1], 1, facecolor='#f0f0ee' if i % 2 == 0 else '#ffffff', edgecolor='none'))
        ax.text(0.1, y + 0.5, row[0], fontsize=9.5, fontweight='bold', va='center')
        for (xv, xt, w), v, t in zip(((xs[1], xs[2], 44), (xs[3], xs[4], 48), (xs[5], xs[6], 78)), row[1::2], row[2::2]):
            if v:
                ax.text(xv, y + 0.5, v, fontsize=8.6, fontweight='bold', va='center', color='#000000')
            ax.text(xt, y + 0.5, '\n'.join(textwrap.wrap(t, w)), fontsize=7.2, va='center', color='#000000', linespacing=1.25)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.5, n + 0.7)
    ax.text(0, -0.35, 'Verdicts by the letter of each run\'s pre-registration. Post-hoc variants never upgrade a verdict. Blank verdict = descriptive, kept, or not scored.', fontsize=8.5, color='#333333')
    fig.savefig('figures/color/grid_three_runs.png', dpi=300, bbox_inches='tight'); print('ok')
    json.dump([dict(cell=r[0], run1=dict(verdict=r[1], note=r[2]), run2=dict(verdict=r[3], note=r[4]), run3=dict(verdict=r[5], note=r[6])) for r in ROWS], open('results/run3/grid_three_runs.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
