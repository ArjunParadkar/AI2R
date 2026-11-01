"""Score P6.3, P7.2, P7.3 from results/run3/sweep_*.json (works on partial sweeps)."""
import json, os, numpy as np
out = {}
for run in ('z113_clean_s0', 'z113_clean_s1', 'z113_scrambled_s0', 'z113_decoy_s1', 'r3_z113_clean_s3', 'r3_z113_clean_s4', 'r3_z113_scrambled_s3', 'r3_z113_scrambled_s4'):
    p = f'results/run3/sweep_{run}.json'
    if not os.path.exists(p):
        continue
    R = json.load(open(p)); te = np.array([r['test_acc'] or 0 for r in R]); grok = int(np.argmax(np.diff(te))) + 1 if len(te) > 1 and te.max() > 0.5 else None
    first = lambda key, cond: next((i for i, r in enumerate(R) if cond(r)), None)
    t_letter = first('order', lambda r: r['order'] >= 113 and r['has_full_cycle']); t_cyc = first('rel', lambda r: r.get('has_full_cycle_rel'))
    t_q = first('q', lambda r: r['q_mid_pullback'] < 0.05 and r['step'] > 0)
    o = dict(n_ckpt=len(R), grok_ckpt=grok, max_order_letter=int(max(r['order'] for r in R)), t_order113_letter=t_letter, t_full_cycle_posthoc=t_cyc,
             n_ckpt_full_cycle_posthoc=int(sum(bool(r.get('has_full_cycle_rel')) for r in R)), all_letter_le2=bool(all(r['order'] <= 2 for r in R)),
             q0=R[0]['q_mid_pullback'], q_max=float(max(r['q_mid_pullback'] for r in R)), t_q_below_005_after0=t_q, q_series=[round(r['q_mid_pullback'], 3) for r in R],
             cycle_series=[int(bool(r.get('has_full_cycle_rel'))) for r in R], order_series=[int(r['order']) for r in R])
    if grok is not None:
        o['posthoc_cycle_offset_from_grok'] = None if t_cyc is None else t_cyc - grok
        after = [r for i, r in enumerate(R) if i >= grok + 2]; o['frac_after_grok_letter_ge113'] = float(np.mean([r['order'] >= 113 for r in after])) if after else None
        o['frac_after_grok_cycle_posthoc'] = float(np.mean([bool(r.get('has_full_cycle_rel')) for r in after])) if after else None
        o['q_swap_first_below_minus_grok'] = None if t_q is None else t_q - grok
    out[run] = o
    print(run, {k: v for k, v in o.items() if not k.endswith('series')})
json.dump(out, open('results/run3/sweep_scores.json', 'w'), indent=1)
