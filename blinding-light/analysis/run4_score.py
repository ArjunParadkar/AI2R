"""Score Run 4 by the letter of SPEC4.md from results/run4/*.json (tolerates missing pieces)."""
import json, os, glob, numpy as np
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
Z, R = J('results/run4/z113_finals.json') or {}, J('results/run4/resnet.json') or {}
out = dict(gate={}, P6={}, P1={}, P7={}, P3={})
cl = [f'r4_z113_clean_s{s}' for s in (5, 6, 7, 8)]; me = [f'r4_z113_scrambled_decoy_s{s}' for s in (5, 6)]; de = [f'r4_z113_decoy_s{s}' for s in (5, 6)]; sh = [f'r4_z113_shortcut_s{s}' for s in (5, 6)]
have = lambda ks: [k for k in ks if k in Z]
# gate
for k in have(me):
    out['gate'][k] = dict(train=Z[k]['train_acc'], test=Z[k]['test_acc'], ok=bool(Z[k]['train_acc'] >= 0.99 and Z[k]['test_acc'] <= 0.02))
for k, v in R.items():
    out['gate'][k] = dict(train=v['train_acc'], test=v['test_acc'], epochs=v['epochs'], ok=bool(v['train_acc'] >= 0.99 and (v['mode'] == 'clean' or v['test_acc'] <= 0.15)))
pairs_z = [(f'r4_z113_clean_s{s}', f'r4_z113_scrambled_decoy_s{s}') for s in (5, 6)]; pairs_r = [(f'r4_resnet_clean_s{s}', f'r4_resnet_scrambled_s{s}') for s in (1, 2)]
def allp(f, pairs, D):
    v = [f(D[a], D[b]) for a, b in pairs if a in D and b in D]; return dict(values=v, n_pairs=len(v))
p61 = dict(z113=allp(lambda c, m: c['P6_1']['C'] - m['P6_1']['C'], pairs_z, Z), resnet=allp(lambda c, m: c['P6_1']['C'] - m['P6_1']['C'], pairs_r, R),
           resnet_per_input_secondary=allp(lambda c, m: c['P6_1']['C_per_input'] - m['P6_1']['C_per_input'], pairs_r, R))
vals = p61['z113']['values'] + p61['resnet']['values']; p61['PASS'] = bool(len(vals) == 4 and all(v > 0.25 for v in vals)); out['P6']['P6_1'] = p61
p62 = dict(clean={k: Z[k]['slot_share'] for k in have(cl)}, mem={k: Z[k]['slot_share'] for k in have(me)})
p62['PASS'] = bool(len(p62['clean']) == 4 and len(p62['mem']) == 2 and all(s[0] + s[1] >= 0.9 and min(s[0], s[1]) >= 0.3 for s in p62['clean'].values()) and all(max(s) <= 0.5 for s in p62['mem'].values())); out['P6']['P6_2'] = p62
p63 = dict(z113=[(Z[a]['q_swap'], Z[b]['q_swap']) for a, b in pairs_z if a in Z and b in Z], resnet=[(R[a]['P6_3']['shift_consistency'], R[b]['P6_3']['shift_consistency']) for a, b in pairs_r if a in R and b in R])
p63['PASS'] = bool(len(p63['z113']) == 2 and len(p63['resnet']) == 2 and all(c < 0.05 and m > 0.3 for c, m in p63['z113']) and all(c >= 0.9 and m <= 0.6 for c, m in p63['resnet'])); out['P6']['P6_3'] = p63
p64 = dict(z113=[(Z[a]['connection']['slope'], Z[b]['connection']['slope']) for a, b in pairs_z if a in Z and b in Z and 'connection' in Z[a] and 'connection' in Z[b]],
           resnet=[(R[a]['P6_4']['slope'], R[b]['P6_4']['slope']) for a, b in pairs_r if a in R and b in R])
allv = p64['z113'] + p64['resnet']; p64['PASS'] = bool(len(allv) == 4 and all(c < -0.3 and m > -0.1 for c, m in allv)); out['P6']['P6_4'] = p64
p65 = dict(clean={k: Z[k]['group']['full_cycle'] for k in have(cl)}, mem={k: Z[k]['group']['full_cycle'] for k in have(me)})
p65['PASS'] = bool(len(p65['clean']) == 4 and len(p65['mem']) == 2 and all(p65['clean'].values()) and not any(p65['mem'].values())); out['P6']['P6_5'] = p65
out['P6']['n_instruments_passed'] = int(sum(out['P6'][k]['PASS'] for k in ('P6_1', 'P6_2', 'P6_3', 'P6_4', 'P6_5'))); out['P6']['BLOCK_PASS'] = bool(out['P6']['n_instruments_passed'] >= 4)
# P1
ch = {k: Z[k].get('characters', {}) for k in have(cl)}
out['P1']['P1_1'] = dict(full_cycle=p65['clean'], mem=p65['mem'], chars={k: dict(n_matched=v.get('n_matched'), n_r09=v.get('n_r09'), unit=v.get('unit'), unit_implied=v.get('unit_implied'), units_4plus=v.get('n_units_4plus')) for k, v in ch.items()})
out['P1']['P1_1']['PASS'] = bool(p65['PASS'] and sum((v.get('n_r09') or 0) >= 4 for v in ch.values()) >= 3)
tw = lambda k: {T: Z[k]['tower'][T]['first_null_layer'] for T in Z[k]['tower']}
out['P1']['P1_3'] = dict(decoy={k: tw(k) for k in have(de)}, mem={k: tw(k) for k in have(me)}, clean={k: tw(k) for k in have(cl)})
ok_d = [ (t['hint_change'] is not None and t['translation'] is not None and t['hint_change'] <= t['translation']) for t in out['P1']['P1_3']['decoy'].values()]
ok_m = [all(v is None for v in t.values()) for t in out['P1']['P1_3']['mem'].values()]
out['P1']['P1_3']['PASS'] = bool(len(ok_d) == 2 and len(ok_m) == 2 and all(ok_d) and all(ok_m))
out['P1']['P1_4'] = dict(clean={k: Z[k]['orbit']['frac_both'] for k in have(cl)}, mem={k: Z[k]['orbit']['frac_both'] for k in have(me)})
out['P1']['P1_4']['PASS'] = bool(len(have(cl)) == 4 and len(have(me)) == 2 and all(v >= 0.99 for v in out['P1']['P1_4']['clean'].values()) and all(v < 0.6 for v in out['P1']['P1_4']['mem'].values()))
# P7
p7 = {}
for k in cl + me + de:
    S = J(f'results/run4/sweep_{k}.json')
    if not S:
        continue
    g = S[0]['grok_ckpt']; cyc = [r for r in S if r['full_cycle']]; first = cyc[0]['ckpt'] if cyc else None; later = [r for r in S if first is not None and r['ckpt'] > first]
    mem_phase = [r for r in S if (r['train_acc'] or 0) >= 0.99 and (r['test_acc'] or 0) < 0.1]
    p7[k] = dict(grok=g, first_cycle=first, offset=None if first is None or g is None else first - g, frac_later=float(np.mean([r['full_cycle'] for r in later])) if later else None, n_later=len(later),
                 n_cycle_ckpts=len(cyc), q0=S[0]['q_swap'], q_max_mem_phase=max([r['q_swap'] for r in mem_phase], default=None), q_below_near_grok=bool(g is not None and any(r['q_swap'] < 0.05 for r in S if abs(r['ckpt'] - g) <= 2)),
                 q_min_after0=min(r['q_swap'] for r in S[1:]) if len(S) > 1 else None, hint_min_mem_phase=min([r['hint_share'] for r in mem_phase], default=None),
                 hint_below_001_near_grok=bool(g is not None and any(r['hint_share'] < 0.01 for r in S if abs(r['ckpt'] - g) <= 2)), n_mem_phase=len(mem_phase))
out['P7']['per_run'] = p7; c7 = [p7[k] for k in cl if k in p7]; m7 = [p7[k] for k in me if k in p7]; d7 = [p7[k] for k in de if k in p7]
out['P7']['P7_1_PASS'] = bool(len(c7) == 4 and len(m7) == 2 and all(r['offset'] is not None and abs(r['offset']) <= 1 and (r['frac_later'] or 0) >= 0.95 for r in c7) and all(r['n_cycle_ckpts'] == 0 for r in m7))
out['P7']['P7_2_PASS'] = bool(len(c7) == 4 and len(m7) == 2 and all(r['q0'] < 0.05 and (r['q_max_mem_phase'] or 0) > 0.2 and r['q_below_near_grok'] for r in c7) and all((r['q_min_after0'] or 0) >= 0.2 for r in m7))
out['P7']['P7_3_PASS'] = bool(len(d7) == 2 and all((r['hint_min_mem_phase'] or 0) > 0.3 and r['hint_below_001_near_grok'] for r in d7))
# P3
out['P3']['P3_1'] = dict(shortcut={k: Z[k]['slot_share'] for k in have(sh)}, clean_max={k: max(Z[k]['slot_share']) for k in have(cl)}, decoy_hint={k: Z[k]['slot_share'][2] for k in have(de)},
                         aonly=Z.get('r4_z113_aonly_s5', {}).get('slot_share'))
p = out['P3']['P3_1']; p['PASS'] = bool(len(p['shortcut']) == 2 and len(p['clean_max']) == 4 and len(p['decoy_hint']) == 2 and p['aonly'] is not None and all(s[2] > 0.8 for s in p['shortcut'].values())
                                        and all(v < 0.6 for v in p['clean_max'].values()) and all(v < 0.01 for v in p['decoy_hint'].values()) and p['aonly'][0] > 0.8)
ints = {}
for s in range(5):
    H = J(f'runs/r4_interrupt_s{s}/history.json')
    if H:
        h = H['history']; by = {x['step']: x for x in h}; ext = [x for x in h if x['step'] > 6000]
        ints[s] = dict(last=h[-1]['step'], first_fire=next((x['step'] for x in h if x['fires']), None), dec6000=by.get(6000, {}).get('dec_acc'), ext_checks=len(ext), ext_fires=int(sum(x['fires'] for x in ext)),
                       test12000=by.get(12000, {}).get('test_acc_in_dist'), max_lam=max(x['lam'] for x in h))
out['P3']['P3_2'] = dict(seeds=ints, PASS=bool(len(ints) == 5 and all((v['dec6000'] or 0) > 0.98 and v['ext_checks'] >= 12 and v['ext_fires'] == 0 and (v['test12000'] or 0) >= 0.99 for v in ints.values())),
                         scored=bool(len(ints) == 5 and all(v['last'] >= 12000 for v in ints.values())))
json.dump(out, open('results/run4/score.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'P7'}, indent=1, default=float)[:6000]); print('P7', json.dumps({k: v for k, v in out['P7'].items() if k != 'per_run'}), json.dumps(out['P7']['per_run'], default=float)[:3000])
