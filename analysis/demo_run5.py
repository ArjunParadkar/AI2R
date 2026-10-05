"""The demo, run 5: open() prints the registered four-instrument verdict (SPEC5 thresholds), the generator and its lead time over the accuracy
jump (from the every-checkpoint sweep), and the FINITE tower.  Live, CPU only, on three models trained after the thresholds were registered."""
import sys, os, glob, time, json
os.environ.setdefault('WB_DEV', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.run5_z113 import *
from whiteboxtopo.symmetry_finite import cycle_coordinate, characters

for name, run, seed, mode, frac in (('generaliser (clean, seed 9)', 'r5_z113_clean_s9', 9, 'clean', .3), ('true memoriser (random labels + decoy hints, seed 9)', 'r5_z113_scrambled_decoy_s9', 9, 'scrambled_decoy', .3),
                                    ('decoy (hint present, uninformative; seed 9)', 'r5_z113_decoy_s9', 9, 'decoy', .5)):
    t0 = time.time(); m = load(sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1], seed); L, r, sig = group(m); s = metric_stats(m, probe(seed, mode, frac)[0][:500])['slot_share']; q = swap_q(m, n=300)['resid_mid']['q_pullback']
    cl, y = clouds_of(run, seed, mode, frac); slope = conn_slope(cl, y)['slope']
    print('=' * 118); print(f'open(Z113 {name})')
    if sig is not None:
        ch = characters(m.W_E[:P].numpy(), cycle_coordinate(sig), top=5); print(f"  (iv) group: generator FOUND (full 113-cycle) = translation by {int(sig[0])}; characters at {[c['freq'] for c in ch]} in the generator's own coordinate")
    else:
        print('  (iv) group: no generator (no full 113-cycle among accepted permutations)')
    und = [s[0] >= 0.40 and s[1] >= 0.40 and s[2] <= 0.05, q < 0.10, slope < -0.35, sig is not None]; mem = [max(s) <= 0.42, q > 0.30, slope > -0.30, sig is None]
    print(f"  (i) slot shares (a, b, hint) = ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})   (ii) q(swap) = {q:.3f}   (iii) depth slope = {slope:+.3f}")
    print(f"  registered verdict: {sum(und)}/4 clauses on the 'understands' side, {sum(mem)}/4 on the 'memorises' side  ->  {'UNDERSTANDS' if all(und) else 'MEMORISES' if all(mem) else 'MIXED'}")
    sw = f'results/run5/sweep_{run}.json'
    if os.path.exists(sw):
        rows = sorted(json.load(open(sw)), key=lambda x: x['ckpt']); g = rows[0]['jump_ckpt']; cyc = [x for x in rows if x['full_cycle']]
        print(f"  formation: " + (f"generator first accepted at checkpoint {cyc[0]['ckpt']} (test accuracy {cyc[0]['test_acc']:.2f}); largest accuracy jump at checkpoint {g}; lead = {g - cyc[0]['ckpt']:+d} checkpoints ({(g - cyc[0]['ckpt']) * 500:+d} steps)" if cyc and g is not None else f"generator never accepted in {len(rows)} checkpoints"))
    tw = finite_tower(m, int(sig[0]) if sig is not None else 1, n=300)
    print('  finite tower q(T) embed -> resid_mid -> resid_post -> logits:  ' + ';  '.join(f"{T} " + ' '.join(f'{tw[T][l]:.2f}' for l in LAYERS) for T in ('swap', 'translation', 'hint_change')) + f'   [{time.time()-t0:.0f}s]')
