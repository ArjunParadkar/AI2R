"""Run 5 scorer: applies SPEC5.md by the letter to results/run5/*.json -> results/run5/score.json.  No threshold lives anywhere but here and in SPEC5."""
import json, os, sys, glob
import numpy as np
from scipy.stats import spearmanr
R = 'results/run5'; LAY = ('embed', 'resid_mid', 'resid_post', 'logits')
CLEAN = ['r5_z113_clean_s%d' % s for s in (9, 10, 11)]; MEM = ['r5_z113_scrambled_decoy_s%d' % s for s in (9, 10)]; DEC = ['r5_z113_decoy_s%d' % s for s in (9, 10)]
RC = ['r5_resnet_cleanaug_s%d' % s for s in (3, 4)]; RM = ['r5_resnet_scrambled_s%d' % s for s in (3, 4)]
lo = lambda f: json.load(open(f'{R}/{f}')) if os.path.exists(f'{R}/{f}') else {}


def gate(Z, RN):
    g = {}
    for k in CLEAN + DEC:
        v = Z.get(k); g[k] = dict(present=v is not None, train=v and v['train_acc'], test=v and v['test_acc'], ok=bool(v and v['train_acc'] >= 0.99 and v['test_acc'] >= 0.99))
    for k in MEM:
        v = Z.get(k); g[k] = dict(present=v is not None, train=v and v['train_acc'], test=v and v['test_acc'], ok=bool(v and v['train_acc'] >= 0.999 and v['test_acc'] <= 0.02))
    for k in RC:
        v = RN.get(k); g[k] = dict(present=v is not None, train=v and v['train_acc'], test=v and v['test_acc'], epochs=v and v['epochs'], ok=bool(v and v['train_acc'] >= 0.99))
    for k in RM:
        v = RN.get(k); g[k] = dict(present=v is not None, train=v and v['train_acc'], test=v and v['test_acc'], epochs=v and v['epochs'], ok=bool(v and v['train_acc'] >= 0.99 and v['test_acc'] <= 0.15))
    return g


def p6(Z, RN, G):
    ok = lambda k: G[k]['ok']; S = {}
    def inst(name, rows):                       # rows: (model, value, passed) ; a gate-failed model fails its clause
        S[name] = dict(models={m: dict(value=v, passed=bool(p)) for m, v, p in rows}, PASS=bool(all(p for _, _, p in rows)))
    inst('P6.a', [(k, Z[k]['q_swap'] if k in Z else None, ok(k) and Z[k]['q_swap'] < 0.10) for k in CLEAN] + [(k, Z[k]['q_swap'] if k in Z else None, ok(k) and Z[k]['q_swap'] > 0.30) for k in MEM])
    inst('P6.b', [(k, RN[k]['P6_b']['shift_consistency'] if k in RN else None, ok(k) and RN[k]['P6_b']['shift_consistency'] >= 0.75) for k in RC] +
                 [(k, RN[k]['P6_b']['shift_consistency'] if k in RN else None, ok(k) and RN[k]['P6_b']['shift_consistency'] <= 0.35) for k in RM])
    sl = lambda k: (Z[k]['connection']['slope'] if k in Z else None) if k.startswith('r5_z113') else (RN[k]['P6_c']['slope'] if k in RN and 'P6_c' in RN[k] else None)
    inst('P6.c', [(k, sl(k), ok(k) and sl(k) is not None and sl(k) < -0.35) for k in CLEAN + RC] + [(k, sl(k), ok(k) and sl(k) is not None and sl(k) > -0.30) for k in MEM + RM])
    ss = lambda k: Z[k]['slot_share'] if k in Z else None
    inst('P6.d', [(k, ss(k), ok(k) and ss(k)[0] >= 0.40 and ss(k)[1] >= 0.40 and ss(k)[2] <= 0.05) for k in CLEAN] + [(k, ss(k), ok(k) and max(ss(k)) <= 0.42) for k in MEM])
    inst('P6.e', [(k, Z[k]['group']['full_cycle'] if k in Z else None, ok(k) and Z[k]['group']['full_cycle']) for k in CLEAN] + [(k, Z[k]['group']['full_cycle'] if k in Z else None, ok(k) and not Z[k]['group']['full_cycle']) for k in MEM])
    S['nulls'] = dict(q_swap_untrained={k: Z[k]['q_swap_untrained'] for k in CLEAN + MEM if k in Z}, shift_untrained={k: RN[k]['P6_b']['untrained'] for k in RC + RM if k in RN},
                      slot_share_untrained={k: Z[k]['slot_share_untrained'] for k in CLEAN + MEM if k in Z}, group_untrained={k: Z[k]['group_untrained']['full_cycle'] for k in CLEAN + MEM if k in Z},
                      slope_label_perm={k: (Z[k]['connection']['null_mean'] if k.startswith('r5_z113') else RN[k]['P6_c']['null_mean']) for k in CLEAN + MEM + RC + RM if (k in Z and 'connection' in Z[k]) or (k in RN and 'P6_c' in RN[k])})
    S['concentration_reported'] = {k: (Z[k]['concentration']['C'] if k in Z else RN[k]['concentration']['C']) for k in CLEAN + MEM + RC + RM if k in Z or (k in RN and 'concentration' in RN[k])}
    S['BLOCK_PASS'] = bool(all(S[i]['PASS'] for i in ('P6.a', 'P6.b', 'P6.c', 'P6.d', 'P6.e'))); return S


def p7(G):
    sw = {k: lo(f'sweep_{k}.json') for k in CLEAN + MEM + DEC}; S = dict(models={})
    for k, rows in sw.items():
        if not rows:
            S['models'][k] = dict(present=False); continue
        rows = sorted(rows, key=lambda r: r['ckpt']); g = rows[0]['jump_ckpt']; cyc = [r['ckpt'] for r in rows if r['full_cycle']]; c1 = cyc[0] if cyc else None; by = {r['ckpt']: r for r in rows}
        d = dict(present=True, n_ckpts=len(rows), jump_ckpt=g, first_accept=c1, lead=(g - c1 if g is not None and c1 is not None else None), test_acc_at_first_accept=(by[c1]['test_acc'] if c1 is not None else None),
                 n_cycle_ckpts=len(cyc))
        if c1 is not None:
            later = [r for r in rows if r['ckpt'] > c1]; d['persistence'] = float(np.mean([r['full_cycle'] for r in later])) if later else None; d['missing_after'] = [r['ckpt'] for r in later if not r['full_cycle']]
        if g is not None:
            near = [r for r in rows if abs(r['ckpt'] - g) <= 3]; d['q_swap_min_within3'] = min(r['q_swap'] for r in near); d['q_swap_at_jump'] = by[g]['q_swap']
            d['q_swap_peak_before_jump'] = max(r['q_swap'] for r in rows if r['ckpt'] <= g); d['has_memorisation_phase'] = any((r['train_acc'] or 0) >= 0.99 and (r['test_acc'] or 0) < 0.1 for r in rows if r['ckpt'] < g)
            d['first_q_below_0.05'] = next((r['ckpt'] for r in rows if r['ckpt'] > 0 and r['q_swap'] < 0.05), None)
        memo = [r for r in rows if (r['train_acc'] or 0) >= 0.99]; d['q_swap_min_train99'] = min((r['q_swap'] for r in memo), default=None)
        if k in DEC and g is not None:
            win = [r for r in rows if (r['train_acc'] or 0) >= 0.99 and r['ckpt'] <= g - 1]; first = win[0]['ckpt'] if win else None; win = [r for r in rows if first is not None and first <= r['ckpt'] <= g - 1]
            d['decline_window'] = [first, g - 1] if win else None; d['decline_n'] = len(win)
            if len(win) >= 5:
                d['decline_spearman'] = float(spearmanr([r['ckpt'] for r in win], [r['hint_share'] for r in win])[0]); d['hint_share_window_start'] = win[0]['hint_share']
            post = [r for r in rows if g <= r['ckpt'] <= g + 4]; d['hint_share_min_g_to_g4'] = min(r['hint_share'] for r in post); d['first_hint_below_0.01'] = next((r['ckpt'] for r in rows if r['ckpt'] > 0 and r['hint_share'] < 0.01), None)
            d['hint_share_final'] = rows[-1]['hint_share']
        S['models'][k] = d
    M = S['models']; full = lambda k: M[k].get('present') and M[k]['n_ckpts'] == 51 and G[k]['ok']
    S['P7.a'] = dict(leads={k: M[k].get('lead') for k in CLEAN}, test_at_accept={k: M[k].get('test_acc_at_first_accept') for k in CLEAN}, mem_cycle_ckpts={k: M[k].get('n_cycle_ckpts') for k in MEM},
                     PASS=bool(all(full(k) and M[k]['lead'] is not None and M[k]['lead'] >= 0 for k in CLEAN) and all(full(k) and M[k]['n_cycle_ckpts'] == 0 for k in MEM)))
    S['P7.b'] = dict(clean_min_within3={k: M[k].get('q_swap_min_within3') for k in CLEAN}, mem_min_train99={k: M[k].get('q_swap_min_train99') for k in MEM}, no_memorisation_phase=[k for k in CLEAN if M[k].get('present') and not M[k].get('has_memorisation_phase')],
                     PASS=bool(all(full(k) and M[k].get('q_swap_min_within3') is not None and M[k]['q_swap_min_within3'] < 0.05 for k in CLEAN) and all(full(k) and M[k]['q_swap_min_train99'] > 0.30 for k in MEM)))
    def dec_ok(k):
        d = M[k]
        if not full(k) or d.get('hint_share_min_g_to_g4') is None or not d['hint_share_min_g_to_g4'] < 0.01:
            return False
        return True if d['decline_n'] < 5 else bool(d['decline_spearman'] <= -0.8 and d['hint_share_window_start'] > 0.30)
    S['P7.c'] = dict(per_decoy={k: {x: M[k].get(x) for x in ('jump_ckpt', 'hint_share_min_g_to_g4', 'first_hint_below_0.01', 'decline_n', 'decline_spearman', 'hint_share_window_start', 'lead', 'persistence')} for k in DEC}, PASS=bool(all(dec_ok(k) for k in DEC)))
    S['P7.d'] = dict(persistence={k: M[k].get('persistence') for k in CLEAN}, missing={k: M[k].get('missing_after') for k in CLEAN}, PASS=bool(all(full(k) and M[k].get('persistence') is not None and M[k]['persistence'] >= 0.95 for k in CLEAN)))
    return S


def tower(Z, G):
    S = {}; tf = lambda k, T: [Z[k]['tower_finite'][T][L] for L in LAY] if k in Z else None
    S['T.1'] = dict(values={k: tf(k, 'hint_change') for k in DEC}, PASS=bool(all(G[k]['ok'] and k in Z and all(v < 0.05 for v in tf(k, 'hint_change')[1:]) for k in DEC)))
    S['T.2'] = dict(values={k: tf(k, 'translation') for k in DEC}, u={k: Z[k]['tower_u'] for k in DEC if k in Z}, PASS=bool(all(G[k]['ok'] and k in Z and all(v > 0.20 for v in tf(k, 'translation')) for k in DEC)))
    S['T.3'] = dict(values={k: tf(k, 'hint_change') for k in MEM}, PASS=bool(all(G[k]['ok'] and k in Z and all(v > 0.15 for v in tf(k, 'hint_change')) for k in MEM)))
    S['BLOCK_PASS'] = bool(S['T.1']['PASS'] and S['T.2']['PASS'] and S['T.3']['PASS']); return S


if __name__ == '__main__':
    Z, RN = lo('z113_finals.json'), lo('resnet.json'); G = gate(Z, RN); out = dict(gate=G)
    if len(sys.argv) < 2 or sys.argv[1] != 'gate':
        out.update(P6=p6(Z, RN, G), P7=p7(G), T=tower(Z, G))
    json.dump(out, open(f'{R}/score.json', 'w'), indent=1, default=float)
    for k, v in G.items():
        print(f"gate {k}: train {v['train']} test {v['test']} -> {'ok' if v['ok'] else 'FAIL'}")
    for b in ('P6', 'P7', 'T'):
        for c, v in out.get(b, {}).items():
            if isinstance(v, dict) and 'PASS' in v:
                print(b, c, 'PASS' if v['PASS'] else 'FAIL')
        if b in out and 'BLOCK_PASS' in out[b]:
            print(b, 'BLOCK', 'PASS' if out[b]['BLOCK_PASS'] else 'FAIL')
