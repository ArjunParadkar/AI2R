"""Run 6 scorer: SPEC6.md + A-15 by the letter -> results/run6/score.json."""
import json, os
import numpy as np
R = 'results/run6'; lo = lambda f: json.load(open(f'{R}/{f}')) if os.path.exists(f'{R}/{f}') else {}
CLEAN = ['r6_z113_clean_s%d' % s for s in (12, 13, 14)]; MEM = ['r6_z113_scrambled_decoy_s%d' % s for s in (15, 16, 17)]
RC = ['r6_resnet_cleanaug_s%d' % s for s in (5, 6)]; RM = ['r6_resnet_scrambled_s%d' % s for s in (5, 6, 7)]


def main():
    Z, RN = lo('z113_finals.json'), lo('resnet.json'); S = {}
    G = {}
    for k in CLEAN: v = Z.get(k); G[k] = bool(v and v['train_acc'] >= 0.99 and v['test_acc'] >= 0.99)
    for k in MEM: v = Z.get(k); G[k] = bool(v and v['train_acc'] >= 0.999 and v['test_acc'] <= 0.02)
    for k in RC: v = RN.get(k); G[k] = bool(v and v['train_acc'] >= 0.99)
    for k in RM: v = RN.get(k); G[k] = bool(v and v['train_acc'] >= 0.99 and v['test_acc'] <= 0.15)
    S['gate'] = G; ok = lambda k: G.get(k, False)
    def inst(name, rows): S[name] = dict(models={m: dict(value=v, passed=bool(p)) for m, v, p in rows}, PASS=bool(rows) and all(p for _, _, p in rows))
    inst('P6.a', [(k, Z[k]['q_swap'] if k in Z else None, ok(k) and Z[k]['q_swap'] < 0.10) for k in CLEAN] + [(k, Z[k]['q_swap'] if k in Z else None, ok(k) and Z[k]['q_swap'] > 0.30) for k in MEM])
    inst('P6.b', [(k, RN[k]['P6_b']['shift_consistency'] if k in RN else None, ok(k) and RN[k]['P6_b']['shift_consistency'] >= 0.75) for k in RC] + [(k, RN[k]['P6_b']['shift_consistency'] if k in RN else None, ok(k) and RN[k]['P6_b']['shift_consistency'] <= 0.35) for k in RM])
    dl = lambda k: (Z[k]['connection']['delta'] if k in Z else None) if k.startswith('r6_z113') else (RN[k]['P6_c']['delta'] if k in RN else None)
    inst('P6.c', [(k, dl(k), ok(k) and dl(k) is not None and dl(k) < -0.25) for k in CLEAN + RC] + [(k, dl(k), ok(k) and dl(k) is not None and dl(k) > -0.10) for k in MEM + RM])
    S['P6.c_transformer_only'] = dict(models={k: dict(value=dl(k)) for k in CLEAN + MEM}, PASS=bool(all(ok(k) and dl(k) is not None and dl(k) < -0.25 for k in CLEAN) and all(ok(k) and dl(k) is not None and dl(k) > -0.10 for k in MEM)))
    ss = lambda k: Z[k]['slot_share'] if k in Z else None; memside = lambda s: s[2] > 0.10 or max(s[0], s[1]) < 0.40
    inst('P6.d', [(k, ss(k), ok(k) and ss(k)[0] >= 0.40 and ss(k)[1] >= 0.40 and ss(k)[2] <= 0.05) for k in CLEAN] + [(k, ss(k), ok(k) and memside(ss(k))) for k in MEM])
    S['P6.d_untrained_null_memoriser_side'] = {k: memside(Z[k]['slot_share_untrained']) for k in CLEAN + MEM if k in Z}
    inst('P6.e', [(k, Z[k]['group']['full_cycle'] if k in Z else None, ok(k) and Z[k]['group']['full_cycle']) for k in CLEAN] + [(k, Z[k]['group']['full_cycle'] if k in Z else None, ok(k) and not Z[k]['group']['full_cycle']) for k in MEM])
    S['P6_BLOCK'] = bool(all(S[i]['PASS'] for i in ('P6.a', 'P6.b', 'P6.c', 'P6.d', 'P6.e')))
    # P7.d at 0.94 from every-checkpoint sweeps
    pers = {}
    for k in CLEAN:
        rows = lo(f'sweep_{k}.json')
        if rows and len(rows) == 51:
            rows = sorted(rows, key=lambda r: r['ckpt']); cyc = [r['ckpt'] for r in rows if r['full_cycle']]; c1 = cyc[0] if cyc else None; later = [r for r in rows if c1 is not None and r['ckpt'] > c1]
            g = rows[0]['jump_ckpt']; pers[k] = dict(first_accept=c1, jump=g, lead=(g - c1 if g is not None and c1 is not None else None), persistence=(float(np.mean([r['full_cycle'] for r in later])) if later else None), missing=[r['ckpt'] for r in later if not r['full_cycle']])
        else:
            pers[k] = dict(persistence=None, n=len(rows))
    S['P7.d'] = dict(models=pers, PASS=bool(all(ok(k) and pers[k].get('persistence') is not None and pers[k]['persistence'] >= 0.94 for k in CLEAN)))
    # K
    k2 = {k: {u: Z[k]['K'][u]['auroc_kappa'] for u in ('oov', 'oov_continuous', 'p97')} for k in CLEAN if k in Z}
    S['K.2'] = dict(auroc_kappa=k2, baselines={k: {u: dict(msp=Z[k]['K'][u]['auroc_msp'], entropy=Z[k]['K'][u]['auroc_entropy'], kappa_s=Z[k]['K'][u]['auroc_kappa_s']) for u in ('oov', 'oov_continuous', 'p97')} for k in CLEAN if k in Z},
                   PASS=bool(all(ok(k) and k in Z and Z[k]['K']['oov']['auroc_kappa'] >= 0.95 and Z[k]['K']['p97']['auroc_kappa'] >= 0.95 for k in CLEAN)),
                   PASS_oov_only=bool(all(ok(k) and k in Z and Z[k]['K']['oov']['auroc_kappa'] >= 0.95 for k in CLEAN)))
    k3 = {}
    for k in RC:
        if k in RN:
            k3[k] = {u: RN[k]['K'][u] for u in ('svhn', 'cifar100', 'noise')}
    beats = lambda d: d['auroc_kappa_s'] > d['auroc_msp'] and d['auroc_kappa_s'] > d['auroc_neg_entropy'] and d['auroc_kappa_s'] > d['auroc_odin'] and d['auroc_kappa_s'] >= d['auroc_golan72']
    S['K.3'] = dict(values=k3, PASS=bool(all(ok(k) and k in RN and beats(RN[k]['K']['svhn']) and beats(RN[k]['K']['cifar100']) for k in RC)),
                   per_cell={k: {u: dict(beats_softmax_family=(RN[k]['K'][u]['auroc_kappa_s'] > max(RN[k]['K'][u]['auroc_msp'], RN[k]['K'][u]['auroc_neg_entropy'], RN[k]['K'][u]['auroc_odin'])), matches_golan=(RN[k]['K'][u]['auroc_kappa_s'] >= RN[k]['K'][u]['auroc_golan72'])) for u in ('svhn', 'cifar100', 'noise')} for k in RC if k in RN})
    tr = {k: (Z[k]['K']['train_kappa_median'] if k in Z else None) for k in CLEAN + MEM}; tr.update({k: (RN[k]['K']['train_kappa_median'] if k in RN else None) for k in RC + RM})
    S['K.4'] = dict(train_kappa_median=tr, PASS=bool(all(ok(k) and tr[k] is not None and tr[k] < 0.3 for k in MEM + RM) and all(ok(k) and tr[k] is not None and tr[k] > 0.95 for k in CLEAN + RC)))
    # F
    fz = {k: (Z[k]['F'] if k in Z else {}) for k in CLEAN}; fr = {k: (RN[k]['F'] if k in RN else {}) for k in RC}; allf = {**fz, **fr}
    def f2(d): return d.get('spearman_eps') is not None and d['spearman_eps'] < 0 and abs(d['spearman_eps']) > abs(d['spearman_margin'])
    S['F.2'] = dict(values={k: {x: d.get(x) for x in ('spearman_eps', 'spearman_eps_bar', 'spearman_margin', 'spearman_msp', 'spearman_neg_kappa_s', 'n_finite')} for k, d in allf.items()},
                   PASS=bool(all(ok(k) and f2(d) for k, d in allf.items()) and len(allf) == 5), PASS_brief_literal=bool(all(ok(k) and d.get('spearman_eps') is not None and d['spearman_eps'] < d['spearman_margin'] for k, d in allf.items()) and len(allf) == 5))
    S['F.2_direction'] = dict(values={k: {x: d.get(x) for x in ('direction_frac_above_0.5', 'direction_frac_above_0', 'direction_cosine_median')} for k, d in fr.items()}, PASS=bool(all(ok(k) and d.get('direction_frac_above_0.5') is not None and d['direction_frac_above_0.5'] >= 0.70 for k, d in fr.items()) and len(fr) == 2), not_applicable='Z113 (token inputs)')
    S['F.3'] = dict(values={k: {x: d.get(x) for x in ('spearman_shuffled_eps', 'spearman_eps_random_transforms', 'eps_random_mean')} for k, d in allf.items()},
                   PASS=bool(all(ok(k) and d.get('spearman_eps_random_transforms') is not None and d['spearman_eps'] < d['spearman_eps_random_transforms'] for k, d in allf.items()) and len(allf) == 5))
    # U
    uz = {k: (Z[k]['U'] if k in Z else {}) for k in CLEAN}; ur = {k: (RN[k]['U'] if k in RN else {}) for k in RC}; allu = {**uz, **ur}
    S['U'] = dict(values={k: d for k, d in allu.items()}, PASS=bool(all(ok(k) and d.get('spearman_kappa_s_eps_bar_pooled') is not None and d['spearman_kappa_s_eps_bar_pooled'] < -0.6 for k, d in allu.items()) and len(allu) == 5),
                  memorisers_vacuous={k: dict(kappa_iqr_train=(Z[k]['U']['kappa_iqr_train'] if k in Z else RN[k]['U']['kappa_iqr_train']) if (k in Z or k in RN) else None) for k in MEM + RM})
    json.dump(S, open(f'{R}/score.json', 'w'), indent=1, default=float)
    for k, v in S.items():
        if isinstance(v, dict) and 'PASS' in v: print(k, 'PASS' if v['PASS'] else 'FAIL')
    print('P6 BLOCK', 'PASS' if S['P6_BLOCK'] else 'FAIL')


if __name__ == '__main__':
    main()
