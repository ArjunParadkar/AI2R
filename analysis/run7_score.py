"""Run 7 scorer: SPEC7.md by the letter -> results/run7/score.json."""
import json, os
import numpy as np
R = 'results/run7'; lo = lambda f: json.load(open(f'{R}/{f}')) if os.path.exists(f'{R}/{f}') else {}
CLEAN_R = ['r6_resnet_cleanaug_s5', 'r6_resnet_cleanaug_s6']; MEM_R = ['r6_resnet_scrambled_s%d' % s for s in (5, 6, 7)]
CLEAN_Z = [f'r{5 if s < 12 else 6}_z113_clean_s{s}' for s in (9, 10, 11, 12, 13, 14)]; MEM_Z = ['r6_z113_scrambled_decoy_s%d' % s for s in (15, 16, 17)]


def main():
    X, XZ, K, P5R, P5Z, RR = lo('X.json'), lo('X_z113.json'), lo('K.json'), lo('P5_resnet.json'), lo('P5_z113.json'), lo('R.json'); S = {}
    # X.4
    lp = lambda k: X[k]['lambda_']['penultimate'] if k in X else None
    i = {k: [f for f in X[k]['learned_penultimate']] for k in CLEAN_R if k in X}; ii = {k: {f: lp(k)[f]['q_untrained'] for f in ('shift_circular', 'hflip')} for k in CLEAN_R if k in X}
    iii = {k: [dict(nearest=r['nearest'], cosine=r['cosine'], lambda_=r['lambda_'], lib_lambda=lp(k)[r['nearest']]['lambda_']) for r in X[k].get('contrastive', [])] for k in CLEAN_R if k in X}
    iv = {k: X[k]['learned_penultimate'] for k in MEM_R if k in X}
    v = {k: dict(swap=XZ[k]['lambda_']['swap']['resid_post'], orbit_translation=XZ[k]['lambda_']['orbit_translation']['resid_post'], sym_translation=XZ[k]['lambda_']['translation']['resid_post']) for k in CLEAN_Z + MEM_Z if k in XZ}
    S['X.4'] = dict(i=i, ii=ii, iii=iii, iv=iv, v=v,
                   i_PASS=bool(all(k in X and len(i[k]) >= 3 for k in CLEAN_R)), ii_PASS=bool(all(k in X and all(np.mean(ii[k][f]) < 0.05 for f in ii[k]) for k in CLEAN_R)),
                   iii_PASS=bool(all(k in X and any(abs(r['cosine']) >= 0.8 and r['lib_lambda'] >= 0.5 for r in iii[k]) for k in CLEAN_R)), iv_PASS=bool(all(k in X and len(iv[k]) < 2 for k in MEM_R)),
                   v_PASS=bool(all(k in XZ and v[k]['orbit_translation'] < 0.2 and v[k]['swap'] >= 0.5 for k in CLEAN_Z) and all(k in XZ and v[k]['swap'] < 0.2 for k in MEM_Z)))
    S['X.4']['PASS'] = all(S['X.4'][p] for p in ('i_PASS', 'ii_PASS', 'iii_PASS', 'iv_PASS', 'v_PASS'))
    # K'.2
    kk = {k: {u: dict(learned=K[k][u]['auroc_kappa_s_learned'], builtin=K[k][u]['auroc_kappa_s_builtin_run6'], msp=K[k][u]['auroc_msp'], entropy=K[k][u]['auroc_entropy'], odin=K[k][u]['auroc_odin'], golan=K[k][u]['auroc_golan72'], nonlearned=K[k][u].get('auroc_kappa_s_nonlearned')) for u in ('svhn', 'cifar100', 'noise')} for k in CLEAN_R if k in K and 'svhn' in K[k]}
    S["K'.2"] = dict(values=kk, untrained_kappa_s={k: K[k].get('untrained_kappa_s_mean') for k in CLEAN_R if k in K}, learned={k: K[k].get('learned') for k in CLEAN_R if k in K},
                    PASS=bool(len(kk) == 2 and all(kk[k][u]['learned'] >= kk[k][u]['builtin'] + 0.20 for k in kk for u in ('svhn', 'cifar100'))),
                    untrained_null_PASS=bool(len(kk) == 2 and all(K[k]['untrained_kappa_s_mean'] < 0.9 for k in kk)),
                    beats_msp={k: {u: kk[k][u]['learned'] > kk[k][u]['msp'] for u in kk[k]} for k in kk})
    S["K'.3"] = {k: dict(frac_misclassified=K[k].get('lowest_frac_misclassified')) for k in CLEAN_R if k in K}
    # P5'
    z = {k: dict(test_acc=v_['test_acc'], auroc_kappa=v_['auroc_kappa'], auroc_kappa_s=v_['auroc_kappa_s'], auroc_msp=v_['auroc_msp'], auroc_entropy=v_['auroc_entropy'], random=v_['auroc_kappa_random_transforms'], generator=v_['generator_found']) for k, v_ in P5Z.items()}
    S["P5'.Z"] = dict(values=z, PASS=bool(len(z) == 3 and all(v_['auroc_kappa'] > v_['auroc_msp'] for v_ in z.values())), null_PASS=bool(len(z) == 3 and all(v_['auroc_kappa'] > v_['random'] for v_ in z.values())))
    r = {k: dict(a=P5R[k]['a'], b={x: P5R[k]['b'][x] for x in P5R[k]['b'] if x != 'cells'}) for k in CLEAN_R if k in P5R}
    S["P5'.R"] = dict(values=r, PASS=bool(len(r) == 2 and all(r[k]['a']['auroc_kappa_s_learned'] > r[k]['a']['auroc_msp'] or r[k]['b']['auroc_kappa_s_learned'] > r[k]['b']['auroc_msp'] for k in r)),
                     null_PASS=bool(len(r) == 2 and all(r[k][c]['auroc_kappa_s_nonlearned'] is not None and r[k][c]['auroc_kappa_s_learned'] > r[k][c]['auroc_kappa_s_nonlearned'] for k in r for c in ('a', 'b'))))
    # R
    rr = {k: dict(rebuilt=RR[k].get('rebuilt_acc'), orig=RR[k].get('original_acc'), same_dir=RR[k].get('n_same_direction'), gamma=RR[k].get('gamma'), var_explained=RR[k].get('logit_variance_explained')) for k in CLEAN_Z if k in RR}
    S['R'] = dict(values=rr, acc_PASS=bool(len(rr) == 6 and all(v_['rebuilt'] is not None and v_['rebuilt'] >= 0.98 for v_ in rr.values())), ablation_PASS=bool(len(rr) == 6 and all(v_['same_dir'] is not None and v_['same_dir'] >= 4 for v_ in rr.values())),
                  gamma_PASS=bool(len(rr) == 6 and all(v_['gamma'] is not None and v_['gamma'] < 0.2 for v_ in rr.values())))
    # P7
    pers = {}
    for k in ('r6_z113_clean_s12', 'r6_z113_clean_s13', 'r6_z113_clean_s14'):
        rows = sorted(json.load(open(f'results/run6/sweep_{k}.json')), key=lambda x: x['ckpt']); cyc = [x['ckpt'] for x in rows if x['full_cycle']]; later = [x for x in rows if x['ckpt'] > cyc[0]]; pers[k] = float(np.mean([x['full_cycle'] for x in later]))
    S['P7'] = dict(persistence=pers, PASS=bool(all(p >= 0.85 for p in pers.values())))
    json.dump(S, open(f'{R}/score.json', 'w'), indent=1, default=float)
    for k, v_ in S.items():
        for p in [x for x in v_ if x.endswith('PASS')] if isinstance(v_, dict) else []: print(k, p, 'PASS' if v_[p] else 'FAIL')


if __name__ == '__main__':
    main()
