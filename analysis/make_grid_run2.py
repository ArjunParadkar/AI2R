"""Both runs side by side, by the letter.  300 dpi, black-and-white safe: status is a word + glyph, fills are greys."""
import sys, os, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.figstyle import *
P, F, H, X = 'PASS', 'FAIL', 'PARTIAL', 'EXPLORATORY'
GL = {P: '✓', F: '✗', H: '◐', X: '?'}
D34 = json.load(open('results/run2/d34_summary.json')) if os.path.exists('results/run2/d34_summary.json') else {}
ROWS = [
 ('P1  what it learned', H, 'Z113 circles 5/5 only with the group parameter; unsupervised none. Ring: FAIL as registered (Euclid 0.75–0.80), post-hoc pullback-geodesic 0.979/0.993/0.993. rot-MNIST FAIL (circle inherited from pixels).',
  F, 'E1 sensitivity-ranked naming: 0/5 on 3 seeds (need ≥4). E2: pullback reduces above-band classes on ring (2→1) and GPT-2 (22→0), not Z113 (12→26). E3: Z₂ quotient MET on 5 models incl. 2 UNSEEN; rot-MNIST invariance depth NOT MET.'),
 ('P6  understands vs memorises', F, 'G jump 0.06/0.16 (need >0.4); scrambled G 0.59 (Z113), 0.89 (ResNet).',
  F, 'A: U_clean − U_scr = 0.000 on ResNet and Z113 (need >0.4): ambient fibers never connect at the registered scale (kill test fires). GW ratio 0.97 (need >2); reversed on Z113.'),
 ('P7  formation', H, 'Births within ±3 ckpts of grokking 12/14; cross-seed order ill-posed.',
  F, 'F: onset mode within ±2 of grokking on 4/5 models; UNSEEN decoy_s1 breaks it (+27), W2 criterion 6/10 pairs, vine-count spread 0.71. Peak lifetime constant (1.01–1.06).'),
 ('P2  why this decision', P, 'IG r = 1.000; margin CV 0.07–0.10. Boundary goes to spectrum-predicted shifts, never c±1.', None, 'kept unchanged'),
 ('P3  bias: detection', P, 'H¹ detector: onset 0–4% late on 5 seeds; 0 FP on clean and decoy nulls. — but EVADED: 0 classes at reliance 1.000.',
  F, 'D2: Mapper alignment threshold = 1.14 (unreachable) → cannot fire. Direct NMI(ŷ,z) floor: fires on 7/7 evasion checkpoints, 0/153 null FPs. Mapper-AMI false-alarms on decoys (44/102).'),
 ('P3  bias: interrupt', H, 'Input-level KL (needs z): 4/5 seeds → 0.998–1.000. Structure-derived penalty: shortcut intact.',
  F, 'D3 by the letter: trigger never fires → no intervention (0.01). ' + D34.get('grid_text', 'POST-HOC direct trigger: running.')),
 ('P4  what it does not know', F, 'd_g loses to softmax everywhere; below chance on GPT-2 (0.40/0.38).',
  F, 'B: ι 0.57/0.59, 1−κ 0.43/0.64, ι(1−κ) 0.50/0.64 vs softmax 0.83/0.82 on SVHN/CIFAR-100. GPT-2 ι 0.82/0.54 (>0.5 MET) but ≡ Euclidean 1-NN (ρ 0.99). Invisible fraction is BELOW chance on near-OOD (0.33).'),
 ('P5  where it will fail', F, 'ρ: Z113 −0.12…−0.23, ResNet +0.32 (wrong sign), GPT-2 null.',
  F, 'C: ρ′ +0.78 on ResNet (need >0.90; softmax margin +0.93), +0.16…+0.19 on Z113 (need >0.5). Beats softmax margin on 0/5 systems. Not a separatrix distance in disguise.'),
 ('m   manifold score', F, 'spectral-gap m: 0.99 ↔ 0.45 on one model; ResNet 0.00.',
  F, 'G: m_plateau CV 0.29 (ring), 0.24 (GPT-2); ResNet penultimate 0.000; ordering wrong. ID(k) profile (plateau/decay/growth) is the informative object.'),
 ('Speed', P, 'open() 4.6 s / 8.3 s.', P, 'run-2 additions +4.4 s (Z113), +2.6 s (ResNet); block instruments 1–130 s except GPT-2 Jacobians (362 s).'),
]


def main():
    setup(); fig, ax = plt.subplots(figsize=(17, 1.05 * len(ROWS) + 1.4)); ax.axis('off')
    xs = [0, 3.5, 4.5, 10.0, 11.0, 17.8]; n = len(ROWS)
    for x, t in ((xs[1], 'RUN 1  (ten-hour spec)'), (xs[3], 'RUN 2  (Morse-theoretic upgrade)')):
        ax.text(x, n + 0.25, t, fontsize=11, fontweight='bold')
    for i, (name, v1, t1, v2, t2) in enumerate(ROWS):
        y = n - 1 - i
        ax.add_patch(plt.Rectangle((0, y), xs[-1], 1, facecolor='#f0f0ee' if i % 2 == 0 else '#ffffff', edgecolor='none'))
        ax.text(0.1, y + 0.5, name, fontsize=10, fontweight='bold', va='center')
        for xv, xt, v, t, w in ((xs[1], xs[2], v1, t1, 62), (xs[3], xs[4], v2, t2, 76)):
            if v:
                ax.text(xv, y + 0.5, f'{GL[v]} {v}', fontsize=9.5, fontweight='bold', va='center', color='#000000')
            ax.text(xt, y + 0.5, '\n'.join(textwrap.wrap(t, w)), fontsize=7.8, va='center', color='#000000', linespacing=1.3)
    ax.set_xlim(0, xs[-1]); ax.set_ylim(-0.5, n + 0.7)
    ax.text(0, -0.35, 'Verdicts by the letter of each run\'s pre-registration (run 1: spec §6.3; run 2: SPEC2.md + AMENDMENTS.md). Post-hoc variants never upgrade a verdict.', fontsize=8.5, color='#333333')
    fig.savefig('figures/grid_both_runs.png', dpi=300, bbox_inches='tight'); print('figures/grid_both_runs.png')
    json.dump([dict(cell=r[0], run1=dict(verdict=r[1], note=r[2]), run2=dict(verdict=r[3], note=r[4])) for r in ROWS], open('results/run2/grid_both_runs.json', 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
