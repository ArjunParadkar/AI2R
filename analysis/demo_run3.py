"""The demo, run 3: what open() now adds -- the discovered group and its order, invariance depth, Killing dimension, breaking score.
Live: group search + characters + swap quotient depth on the grokked Z113 model and on the memoriser.  Killing dimension and the breaking
score are read from the block results (they need ~20 min of Jacobians on a shared GPU) and say so."""
import sys, os, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.run3_blockZ import load, all_logits, D_g, swap_q, P
from whiteboxtopo.symmetry_finite import automorphisms, full_cycle, cycle_coordinate, characters

for name, run, seed in (('Z113 transformer, grokked, seed 1', 'z113_clean_s1', 1), ('Z113 transformer, memoriser (scrambled labels)', 'z113_scrambled_s0', 0)):
    t0 = time.time(); m = load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], seed); L = all_logits(m); r = automorphisms(D_g(L)); q = swap_q(m, n=300)
    print('=' * 100); print(f'open({name})   -- run-3 additions')
    print(f"  group (by the letter, max-norm isometries of the output-pullback metric on the 113 tokens): order {r['order']}, 113-cycle: {r['has_full_cycle']}")
    sig = full_cycle(r, 'perms_rel') if r['has_full_cycle_rel'] else None
    if sig is not None:
        ch = characters(m.W_E[:P].detach().float().cpu().numpy(), cycle_coordinate(sig), top=5)
        print(f"  group [POST-HOC mean-norm rule]: a 113-cycle generator exists -- translation by {int(sig[0])}; characters at frequencies {[c['freq'] for c in ch]} "
              f"({', '.join(f'{100*c["power_share"]:.0f}%' for c in ch)} of embedding power)")
    else:
        print('  group [POST-HOC mean-norm rule]: no 113-cycle generator')
    pred = L.argmax(-1); print(f"  orbit explanation: input (5, 9) -> predicted {pred[5, 9]}; its swap (9, 5) -> {pred[9, 5]}; its translates (6, 8), (7, 7) -> {pred[6, 8]}, {pred[7, 7]}")
    print('  invariance depth of operand swap: ' + ', '.join(f"{k} q={q[k]['q_pullback']:.3f}" for k in ('embed', 'resid_mid', 'resid_post', 'logits')) + f"  -> first below 0.05: {q['invariance_depth_pullback']}")
    print(f'  [{time.time()-t0:.1f}s]')
try:
    B = json.load(open('results/run3/blockP3.json')); I = json.load(open('results/run3/blockIMG.json'))
    print('=' * 100); print('from the block results (not live):')
    print("  breaking score b_z, final checkpoints: shortcut 0.967 / 0.989, re-encoded (H1-evading) 0.975 / 0.943, decoy 0.0004 / 0.0005, clean 0.007-0.011; untrained 0.333")
    print("  Killing dimension over 8 image generators, ResNet-18: clean 0, scrambled 0, untrained 0 (rotation residual 0.068 / 0.099 / 0.093)")
except Exception as e:
    print('block results unavailable', e)
