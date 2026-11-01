"""The demo, run 4: open() prints the group and its generator, the tower, the P6 instruments with a memorisation read-out, and the
dominant-direction flag.  Live, CPU only, on three fresh models: a generaliser, a true memoriser, a planted-shortcut model."""
import sys, os, glob, time
os.environ.setdefault('WB_DEV', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.run4_z113 import *

for name, run, seed, mode, frac in (('fresh generaliser (clean, seed 7)', 'r4_z113_clean_s7', 7, 'clean', .3), ('true memoriser (random labels + decoy hints, seed 5)', 'r4_z113_scrambled_decoy_s5', 5, 'scrambled_decoy', .3),
                                    ('planted-shortcut model (seed 5)', 'r4_z113_shortcut_s5', 5, 'shortcut', .5)):
    t0 = time.time(); m = load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], seed); L, r, sig = group(m); ms = metric_stats(m, probe(seed, mode, frac)[0][:500]); q = swap_q(m, n=300)['resid_mid']['q_pullback']
    print('=' * 110); print(f'open(Z113 {name})')
    if sig is not None:
        ch = characters(m.W_E[:P].numpy(), cycle_coordinate(sig), top=5)
        print(f"  group: generator FOUND (statistical symmetry, full 113-cycle) = translation by {int(sig[0])}; characters at {[c['freq'] for c in ch]} in the generator's own coordinate")
    else:
        print('  group: no generator (no full 113-cycle among accepted permutations)')
    tw = tower(m, int(sig[0]) if sig is not None else 1, n=300)
    print('  tower  q(T) embed -> resid_mid -> resid_post -> logits:  ' + ';  '.join(f"{T} " + ' '.join(f'{tw[T][l]:.2f}' for l in ('embed', 'resid_mid', 'resid_post', 'logits')) for T in tw))
    s = ms['slot_share']; votes = [max(s) <= 0.5 and min(s) > 0.1, q > 0.3, sig is None]
    print(f"  instruments: slot shares (a, b, hint) = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f});  q(swap) = {q:.3f};  generator {'yes' if sig is not None else 'no'}   ->  memorisation read-out: {sum(votes)}/3 instruments on the memoriser side")
    flag = [n for n, v in zip(('a', 'b', 'hint'), s) if v > 0.8]
    print(f"  dominant-direction flag: {'slot ' + flag[0] + f' holds {max(s):.0%} of the input geometry -- check it against the task' if flag else 'none'}   [{time.time()-t0:.0f}s]")
