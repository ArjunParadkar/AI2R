"""Phase 2 analysis — implements prereg/PHASE_2.md exactly.  python analysis/phase2_analyze.py [resultsdir] [out.json]
Unit of analysis = one independent training run (seed).  Authors: Arjun Paradkar and Reva Paradkar."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import stats as S

RES = sys.argv[1] if len(sys.argv) > 1 else 'results/phase2'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'results/phase2_summary.json'
MOD = ['mod59', 'mod97', 'mod113', 'comp60', 'comp96']; NONAB = ['S5', 'A5', 'D12']
BASE = ['base_wnorm', 'base_entropy', 'base_msp', 'base_margin', 'base_probe_loss', 'base_hess_trace']   # trivial baselines (sign-agnostic: |AUROC-.5|)


def load():
    rows = []
    for f in glob.glob(f'{RES}/*.json'):
        for r in json.load(open(f)):
            r['cfg_id'] = os.path.basename(f)[:-5]; rows.append(r)
    return rows


def hp_of(r): return json.loads(r['hp']) if isinstance(r.get('hp'), str) else r.get('hp', {})
def alpha_of(r): return hp_of(r).get('model', {}).get('alpha', 1.0)
def sel(rows, **kw): return [r for r in rows if all((r.get(k) == v) for k, v in kw.items())]
def col(rows, k): return np.array([r[k] if r.get(k) is not None else np.nan for r in rows], float)
def understands(r): return (r.get('kappa', 0) >= 0.8) and (r.get('q_logit', 9) <= 0.5)
def best_sign_auroc(pos, neg): a = S.auroc(pos, neg); return max(a, 1 - a)


def main():
    rows = load(); R = {'n_rows': len(rows)}; pv = {}
    fin = [r for r in rows if r['snapshot'] == 'final']
    # ---- H2.1 clock vs pizza: D1 separates alpha=0 (pizza) from alpha=1 (clock) among models with test_acc >= .99 -------------------------
    h21 = {}
    for t in ('mod59', 'mod97', 'mod113'):
        c = [r for r in sel(fin, task=t, arch='tf1', regime='clean') if r['test_acc'] >= 0.99]
        pz, ck = [r for r in c if alpha_of(r) == 0.0], [r for r in c if alpha_of(r) == 1.0]
        if len(pz) < 5 or len(ck) < 5: h21[t] = dict(testable=False, n_pizza=len(pz), n_clock=len(ck)); continue
        d = {}
        for name in ('swap_asym', 'd1_eps', 'd1_ampcv') + tuple(BASE) + ('fourier_top5', 'kappa', 'q_logit'):
            a, p, kind = S.perm_p_auroc(col(ck, name), col(pz, name)); d[name] = dict(auroc_clock_gt_pizza=a, best_sign=max(a, 1 - a), p=p, test=kind)
        a, lo, hi = S.boot_ci_auroc(col(ck, 'swap_asym'), col(pz, 'swap_asym'))
        d['headline'] = dict(descriptor='swap_asym', auroc=a, ci=[lo, hi], p=d['swap_asym']['p'], threshold=0.9, direction='clock(alpha=1) > pizza(alpha=0) (pre-registered, from pilot seeds 910-925 and mechanism: uniform attention is permutation-symmetric)', pass_=bool(a >= 0.9),
                             caveat='descriptor largely reads the architecture (alpha) that also defines the label')
        d['n_pizza'], d['n_clock'] = len(pz), len(ck); d['testable'] = True; h21[t] = d; pv[f'H2.1/{t}'] = d['swap_asym']['p']
    R['H2.1'] = h21
    # ---- H2.2 hybrids: per-seed AUROC of (1-kappa) for exception vs rule on training inputs -------------------------------------------------
    h22 = {}
    for t in ('mod59', 'mod97', 'mod113'):
        c = [r for r in sel(fin, task=t, arch='tf1', regime='hybrid') if r.get('exc_memorised', 0) >= 0.95 and r['train_acc'] >= 0.99]
        if len(c) < 5: h22[t] = dict(testable=False, n=len(c)); continue
        a = col(c, 'auroc_exc_inc_kappa'); m, lo, hi = S.boot_ci(a)
        bl = {k: S.boot_ci(col(c, f'auroc_exc_{k}'))[0] for k in ('eps_bar', 'inc_kappa_s', 'one_minus_msp', 'entropy', 'neg_margin', 'loss')}
        h22[t] = dict(testable=True, n=len(c), mean=m, ci=[lo, hi], pass_=bool(m >= 0.9 and lo >= 0.85), baselines_mean_auroc=bl, n_rule_learned_te50=int(sum(r['test_acc'] >= 0.5 for r in c)),
                      best_baseline=max(bl.values()), beats_baselines=bool(m > max(bl.values())))
    R['H2.2'] = h22
    # ---- H2.3 nograk / contaminated classified non-generalising; clean grokked classified generalising ---------------------------------------
    h23 = {}
    for t in MOD:
        d = {}
        for reg in ('clean', 'nograk', 'contaminated', 'scrambled'):
            c = sel(fin, task=t, arch='tf1', regime=reg)
            if reg == 'clean': c = [r for r in c if r['test_acc'] >= 0.99]
            if reg == 'nograk': c = [r for r in c if r['test_acc'] < 0.5 and r['train_acc'] >= 0.99]
            d[reg] = dict(n=len(c), frac_understands=float(np.mean([understands(r) for r in c])) if c else None)
        ok = (d['clean']['n'] >= 5 and d['clean']['frac_understands'] is not None and d['clean']['frac_understands'] >= 0.9 and
              all(d[k]['n'] >= 5 and d[k]['frac_understands'] <= 0.1 for k in ('nograk', 'contaminated')))
        d['pass_'] = bool(ok); h23[t] = d
    R['H2.3'] = h23
    # ---- H2.4 non-abelian blind group recovery ---------------------------------------------------------------------------------------------
    h24 = {}
    for t in NONAB:
        cl = [r for r in sel(fin, task=t, arch='tf2', regime='clean') if r['test_acc'] >= 0.99]; sc = sel(fin, task=t, arch='tf2', regime='scrambled')
        h24[t] = dict(n_clean=len(cl), recovered_clean=int(sum(bool(r.get('group_recovered')) for r in cl)), n_scrambled=len(sc),
                      recovered_scrambled=int(sum(bool(r.get('group_recovered')) for r in sc)), pred_clean=[r.get('group_pred') for r in cl])
        h24[t]['pass_'] = bool(len(cl) >= 8 and h24[t]['recovered_clean'] >= 8 and h24[t]['recovered_scrambled'] <= 1)
    R['H2.4'] = h24
    # ---- H2.6 trivial baselines vs the structure instruments on the three contrasts -------------------------------------------------------------
    c2 = {}
    for t in MOD:
        g_ = [r for r in sel(fin, task=t, arch='tf1', regime='clean') if r['test_acc'] >= 0.99 and alpha_of(r) == 1.0]; ng_ = [r for r in sel(fin, task=t, arch='tf1', regime='nograk') if r['test_acc'] < 0.5 and r['train_acc'] >= 0.99]
        if len(g_) >= 5 and len(ng_) >= 5:
            wb = best_sign_auroc(col(g_, 'kappa'), col(ng_, 'kappa')); bl = {k: best_sign_auroc(col(g_, k), col(ng_, k)) for k in BASE}; c2[t] = dict(wb=wb, best_baseline=max(bl.values()), baselines=bl)
    c3 = {t: dict(wb=h21[t]['headline']['auroc'], best_baseline=max(max(h21[t][k]['best_sign'] for k in BASE), 0)) for t in h21 if h21[t].get('testable')}
    c1 = {t: dict(wb=h22[t]['mean'], best_baseline=h22[t]['best_baseline']) for t in h22 if h22[t].get('testable')}
    def contrast(c, k_pass):
        if not c: return dict(testable=False)
        wbp = sum(v['wb'] >= 0.9 for v in c.values()); blf = sum(v['best_baseline'] < 0.9 for v in c.values()); n = len(c)
        return dict(per_task=c, wb_pass_count=wbp, baseline_fail_count=blf, n=n, wb_passes=bool(wbp >= k_pass(n)), baseline_fails=bool(blf >= k_pass(n)))
    h26 = dict(C1_hybrid_exceptions=contrast(c1, lambda n: max(2, int(np.ceil(2 * n / 3)))), C2_nograk_vs_generalizer=contrast(c2, lambda n: int(np.ceil(0.8 * n))), C3_clock_pizza=contrast(c3, lambda n: max(2, int(np.ceil(2 * n / 3)))))
    h26['claim_holds'] = bool(any(v.get('wb_passes') and v.get('baseline_fails') for v in h26.values() if isinstance(v, dict) and 'wb_passes' in v))
    R['H2.6'] = h26
    # ---- H2.5 symmetry-free task ----------------------------------------------------------------------------------------------------------------
    cl = sel(fin, task='symfree', arch='tf2', regime='clean'); sc = sel(fin, task='symfree', arch='tf2', regime='scrambled')
    if len(cl) >= 8 and len(sc) >= 8:
        a, p, kind = S.perm_p_auroc(col(cl, 'kappa_local'), col(sc, 'kappa_local')); pv['H2.5/kappa_local'] = p
        R['H2.5'] = dict(n_clean=len(cl), generalises_frac=float(np.mean([r['test_acc'] >= 0.7 for r in cl])), no_global_cycle_frac=float(np.mean([not r.get('auto_full_cycle', False) for r in cl])),
                         auroc_kappa_local=a, p=p, pass_=bool(np.mean([r['test_acc'] >= 0.7 for r in cl]) >= 0.8 and np.mean([not r.get('auto_full_cycle', False) for r in cl]) >= 0.9 and a >= 0.9))
    else: R['H2.5'] = dict(testable=False, n_clean=len(cl), n_scrambled=len(sc))
    # ---- H2.7 covering number vs generalisation gap (modular tasks, regimes clean/scrambled/hybrid/nograk + midgrok snapshots; cluster bootstrap over seeds) -----------
    pool = [r for r in rows if r['task'] in MOD and r['arch'] == 'tf1' and r['regime'] in ('clean', 'scrambled', 'hybrid', 'nograk', 'midgrok') and alpha_of(r) == 1.0]
    if len(pool) >= 60:
        gap = col(pool, 'train_acc') - col(pool, 'test_acc'); seeds = np.array([r['seed'] for r in pool]); out = {}
        for k in ('cover_r0.5', 'base_wnorm', 'base_hess_trace', 'base_entropy', 'base_probe_loss'):
            out[k] = S.spearman_boot(col(pool, k), gap)
        rng = np.random.default_rng(0); us = np.unique(seeds); diffs = {b: [] for b in ('base_wnorm', 'base_hess_trace')}
        from scipy.stats import spearmanr
        for _ in range(2000):
            ss = rng.choice(us, len(us)); idx = np.concatenate([np.where(seeds == s_)[0] for s_ in ss])
            for b in diffs:
                x, y, g = col(pool, 'cover_r0.5')[idx], col(pool, b)[idx], gap[idx]; ok = ~(np.isnan(x) | np.isnan(y))
                diffs[b].append(abs(spearmanr(x[ok], g[ok]).statistic) - abs(spearmanr(y[ok], g[ok]).statistic))
        R['H2.7'] = dict(n=len(pool), spearman={k: dict(rho=v[0], ci=[v[1], v[2]]) for k, v in out.items()},
                         paired_abs_diff_vs={b: dict(mean=float(np.nanmean(d)), ci=[float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]) for b, d in diffs.items()},
                         pass_=bool(abs(out['cover_r0.5'][0]) >= 0.7 and all(np.nanquantile(d, .025) > 0 for d in diffs.values())))
    else: R['H2.7'] = dict(testable=False, n=len(pool))
    # ---- H-symp ---------------------------------------------------------------------------------------------------------------------------------
    hs = []
    for r in sel(fin, regime='clean'):
        if r['task'] not in MOD or r['arch'] != 'tf1' or alpha_of(r) != 1.0 or r['test_acc'] < 0.99: continue
        tr = r.get('traj') or []; ks = [t for t in tr if 'kappa_probe' in t and 'J_attn' in t]
        if len(ks) < 8: continue
        step = np.array([t['step'] for t in ks]); kap = np.array([t['kappa_probe'] for t in ks]); J = np.array([t['J_attn'] + 1e-12 for t in ks]); W = np.array([t['wnorm'] for t in ks])
        idx = np.where((kap >= 0.8) & (np.r_[kap[1:], 1.0] >= 0.8))[0]
        if len(idx) == 0: continue
        fk = int(idx[0]); tJ = int(np.argmax(np.abs(np.diff(np.log(J))))); tW = int(np.argmax(np.abs(np.diff(np.log(W)))))
        hs.append(dict(task=r['task'], seed=r['seed'], f_kappa=fk, t_J=tJ, t_W=tW, lead_J=fk - tJ, lead_W=fk - tW))
    if len(hs) >= 8:
        lj = np.array([h['lead_J'] for h in hs]); lw = np.array([h['lead_W'] for h in hs]); frac = float(np.mean(lj >= -1)); m, lo, hi = S.boot_ci(lj - lw)
        R['H-symp'] = dict(n=len(hs), frac_J_precedes_or_coincides=frac, lead_diff_J_minus_W=dict(mean=m, ci=[lo, hi]), pass_=bool(frac >= 0.8 and lo > 0), kill=bool(frac < 0.8 or lo <= 0))
    else: R['H-symp'] = dict(testable=False, n=len(hs))
    # ---- M5 causal block --------------------------------------------------------------------------------------------------------------------------
    m5 = {}
    for t, a in (('mod59', 'tf1'), ('mod113', 'tf1'), ('S5', 'mlp2'), ('permcomp5', 'mlp2')):
        base = sel(fin, task=t, arch=a, regime='clean'); ng = sel(fin, task=t, arch=a, regime='nograk'); d = {}
        for reg in ('destroy', 'induce', 'transplant'):
            c = sel(fin, task=t, arch=a, regime=reg); d[reg] = dict(n=len(c), test_acc_mean=S.boot_ci(col(c, 'test_acc')) if c else None)
        d['clean'] = dict(n=len(base), test_acc_mean=S.boot_ci(col(base, 'test_acc')) if base else None); d['nograk'] = dict(n=len(ng), test_acc_mean=S.boot_ci(col(ng, 'test_acc')) if ng else None)
        if len(base) >= 5 and d['destroy']['n'] >= 5:
            diff = col(base, 'test_acc').mean() - col(sel(fin, task=t, arch=a, regime='destroy'), 'test_acc').mean(); d['destroy_drop'] = float(diff)
            d['destroy_pass'] = bool(diff >= 0.2 and S.perm_p_auroc(col(base, 'test_acc'), col(sel(fin, task=t, arch=a, regime='destroy'), 'test_acc'))[1] < 0.05)
        for reg in ('induce', 'transplant'):
            c = sel(fin, task=t, arch=a, regime=reg)
            if len(c) >= 5: d[f'{reg}_pass'] = bool(np.mean([r['test_acc'] >= 0.9 for r in c]) >= 0.6 and (not ng or np.mean([r['test_acc'] >= 0.9 for r in ng]) < 0.2))
        m5[f'{t}/{a}'] = d
    R['M5'] = m5
    json.dump(R, open(OUT, 'w'), indent=1, default=float); print(json.dumps(R, indent=1, default=float)[:3000])


if __name__ == '__main__': main()
