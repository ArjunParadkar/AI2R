"""Run 3, Block P3 (SPEC3.md S5 + P3.1-P3.4): spurious structure as symmetry breaking.
b_z = E_x ||J_hint||^2_F / ||J||^2_F, J = d logits / d (a, b, hint slot embeddings), on the run-2 decorrelated probe.
Stages (--only): p31 (b_z on every checkpoint of the nine runs + threshold), score31, p32, p34, fig, score33."""
import sys, os, re, json, glob, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, P, NULL, EQ
from whiteboxtopo.symmetry_breaking import breaking_score

OUT = 'results/run3/blockP3.json'
RUNS = [('z113_clean_s0', 0, 'clean', .3), ('z113_clean_s1', 1, 'clean', .3), ('z113_cleanctrl_s2', 2, 'clean', .5),
        ('z113_decoy_s0', 0, 'decoy', .5), ('z113_decoy_s1', 1, 'decoy', .5),
        ('z113_shortcut_s0', 0, 'shortcut', .5), ('z113_shortcut_s1', 1, 'shortcut', .5),
        ('r2_reencoded_s0', 0, 'shortcut', .5), ('r2_reencoded_s1', 1, 'shortcut', .5)]
CAL = ('z113_clean_s0', 'z113_clean_s1')
CLEAN = CAL + ('z113_cleanctrl_s2',); DECOY = ('z113_decoy_s0', 'z113_decoy_s1')


def slot_fn(model, layers=False):
    """One-example function of the three input slots (3,128) [token + positional embedding at positions 0,1,2]; the EQ slot is fixed."""
    eq = (model.W_E[EQ] + model.W_pos[3]).detach()

    def f(slots):
        x = torch.cat([slots, eq[None]], 0)[None]
        x = x + model.attn(x)
        mid = x[0, -1]
        post = mid + model.mlp(mid)
        logits = post @ model.W_U
        return (mid, post, logits) if layers else logits
    return f


def slots_of(model, tk):
    with torch.no_grad():
        return model.embed(torch.as_tensor(tk, device=model.W_E.device))[:, :3].detach().clone()


def bz(model, tk, layers=False, metric=False):
    was = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        r = breaking_score(slot_fn(model, layers), slots_of(model, tk), nuisance=2, batch=BATCH, return_metric=metric,
                           names=['resid_mid', 'resid_post', 'logits'] if layers else None)
    finally:
        for p, w in zip(model.parameters(), was):
            p.requires_grad_(w)
    return r


DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 250 if DEV == 'cuda' else 50


def load(ck, seed):
    m = Z113Transformer(seed=seed).to(DEV); m.load_state_dict(torch.load(ck, map_location=DEV)); m.eval(); return m


def clean(r):
    return {k: v for k, v in r.items() if k not in ('ratios', 'metric')}


def get_probe(seed, mode, frac):
    from analysis.run2_blockD12 import probe
    return probe(seed, mode, frac)


def jload():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def jsave(o):
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT + '.tmp', 'w'), indent=1, default=float); os.replace(OUT + '.tmp', OUT)


def p31(only_runs=None):
    """With --runs: compute those runs only and write results/run3/p31_parts/<run>.json (merged by merge31)."""
    t0 = time.time(); o = jload() if only_runs is None else {}; o['runs'] = {}
    floor = json.load(open('results/run2/blockD12.json'))
    for run, seed, mode, frac in RUNS:
        if only_runs is not None and run not in only_runs:
            continue
        tk, z, y = get_probe(seed, mode, frac); fl = {r['step']: r for r in floor[run]}; rows = []
        for ck in sorted(glob.glob(f'runs/{run}/ckpt_*.pt')):
            step = int(re.search(r'ckpt_(\d+)', ck)[1]); m = load(ck, seed); r = bz(m, tk)
            q = np.quantile(r['ratios'].numpy(), [.05, .5, .95])
            with torch.no_grad():
                dec = float((m(torch.as_tensor(tk, device=DEV)).argmax(-1).cpu().numpy() == y).mean())
            f = fl[step]
            rows.append(dict(step=step, b_z=r['b_z'], b_z_rom=r['b_z_rom'], slot_share=r['slot_share'], q05=q[0], q50=q[1], q95=q[2], n_zero=r['n_zero'], dec_acc_probe=dec,
                             floor=f['direct_nmi'], floor_fires=f['fires_direct'], **{k: f[k] for k in ('h1_spurious', 'reliance') if k in f}))
        o['runs'][run] = rows; b = np.array([r['b_z'] for r in rows])
        print(f'[P3.1] {run:20s} n={len(b)} b_z mean {b.mean():.4f} sd {b.std():.4f} min {b.min():.4f} max {b.max():.4f} final {b[-1]:.4f}  ({time.time()-t0:.0f}s)', flush=True)
        if only_runs is not None:
            os.makedirs('results/run3/p31_parts', exist_ok=True)
            json.dump(dict(rows=rows, device=DEV, wall_s=time.time() - t0), open(f'results/run3/p31_parts/{run}.json', 'w'), indent=1, default=float); continue
        if run == CAL[-1]:      # threshold FIRST (P3.3 needs it)
            cal = np.array([r['b_z'] for k in CAL for r in o['runs'][k]])
            o['threshold'] = dict(threshold=float(cal.mean() + 3 * cal.std()), cal_mean=float(cal.mean()), cal_sd=float(cal.std()), n_cal=len(cal), sd='population (np.std, ddof=0), as run 2',
                                  threshold_floor=floor['_verdict']['threshold_direct'])
            print('[P3.1] threshold', o['threshold'], flush=True); jsave(o)
    if only_runs is None:
        o['timing_p31_s'] = time.time() - t0; jsave(o)


def merge31():
    o = jload(); o['p31_devices'] = {k: 'cuda' for k in o['runs']}; o['timing_p31_parts_s'] = {}
    for run, *_ in RUNS:
        p = f'results/run3/p31_parts/{run}.json'
        if os.path.exists(p) and run not in CAL:
            d = json.load(open(p)); o['runs'][run] = d['rows']; o['p31_devices'][run] = d['device']; o['timing_p31_parts_s'][run] = d['wall_s']
    o['runs'] = {run: o['runs'][run] for run, *_ in RUNS if run in o['runs']}
    print({k: len(v) for k, v in o['runs'].items()}); jsave(o)


def score31():
    from scipy.stats import spearmanr
    o = jload(); thr = o['threshold']['threshold']; R = o['runs']
    for rows in R.values():
        for r in rows:
            r['fires'] = bool(r['b_z'] > thr)
    allr = [(k, r) for k, rows in R.items() for r in rows]
    b = np.array([r['b_z'] for _, r in allr]); f = np.array([r['floor'] for _, r in allr]); brom = np.array([r['b_z_rom'] for _, r in allr])
    ff = np.array([r['floor_fires'] for _, r in allr]); bf = np.array([r['fires'] for _, r in allr])
    rho = float(spearmanr(b, f)[0]); rho_rom = float(spearmanr(brom, f)[0])
    grp = lambda keys: [r for k in keys for r in R[k]]
    ev = [(k, r) for k in ('r2_reencoded_s0', 'r2_reencoded_s1') for r in R[k] if r.get('h1_spurious') == 0 and r.get('reliance', 0) > 0.9]
    per_run = {k: dict(n=len(rows), b_fires=int(sum(r['fires'] for r in rows)), floor_fires=int(sum(r['floor_fires'] for r in rows)),
                       floor_fires_b_silent=[r['step'] for r in rows if r['floor_fires'] and not r['fires']],
                       b_fires_floor_silent=[r['step'] for r in rows if r['fires'] and not r['floor_fires']],
                       spearman_within=float(spearmanr([r['b_z'] for r in rows], [r['floor'] for r in rows])[0]),
                       b_z_mean=float(np.mean([r['b_z'] for r in rows])), b_z_max=float(np.max([r['b_z'] for r in rows])), b_z_final=rows[-1]['b_z'],
                       floor_max=float(np.max([r['floor'] for r in rows]))) for k, rows in R.items()}
    sub = lambda keys: float(spearmanr([r['b_z'] for r in grp(keys)], [r['floor'] for r in grp(keys)])[0])
    v = dict(n_pooled=len(b), spearman_pooled=rho, spearman_pooled_rom=rho_rom, spearman_MET=bool(rho > 0.9),
             spearman_planted_runs_only=sub(('z113_shortcut_s0', 'z113_shortcut_s1', 'r2_reencoded_s0', 'r2_reencoded_s1')),
             spearman_null_runs_only=sub(CLEAN + DECOY),
             floor_fires_total=int(ff.sum()), b_fires_where_floor_fires=int((ff & bf).sum()), floor_fires_b_silent=int((ff & ~bf).sum()), b_fires_floor_silent=int((bf & ~ff).sum()),
             fires_wherever_floor_fires_MET=bool((ff & ~bf).sum() == 0),
             evasion=[dict(run=k, step=r['step'], b_z=r['b_z'], fires=r['fires'], floor=r['floor'], floor_fires=r['floor_fires']) for k, r in ev],
             evasion_n=len(ev), evasion_all_fire_MET=bool(len(ev) == 7 and all(r['fires'] for _, r in ev)),
             fp_clean_cal=f"{sum(r['fires'] for r in grp(CAL))}/{len(grp(CAL))}", fp_clean_heldout_s2=f"{sum(r['fires'] for r in R['z113_cleanctrl_s2'])}/{len(R['z113_cleanctrl_s2'])}",
             fp_clean_total=int(sum(r['fires'] for r in grp(CLEAN))), n_clean=len(grp(CLEAN)),
             fp_decoy_total=int(sum(r['fires'] for r in grp(DECOY))), n_decoy=len(grp(DECOY)),
             fp_decoy_steps={k: [r['step'] for r in R[k] if r['fires']] for k in DECOY},
             fp_clean_steps={k: [r['step'] for r in R[k] if r['fires']] for k in CLEAN},
             per_run=per_run)
    v['zero_false_firings_MET'] = bool(v['fp_clean_total'] == 0 and v['fp_decoy_total'] == 0)
    v['P3_1_PASS'] = bool(v['spearman_MET'] and v['fires_wherever_floor_fires_MET'] and v['evasion_all_fire_MET'] and v['zero_false_firings_MET'])
    o['P3_1'] = v; jsave(o); print(json.dumps(v, indent=1, default=float))


FINALS = [('z113_shortcut_s0', 0, 'shortcut', .5), ('z113_shortcut_s1', 1, 'shortcut', .5), ('r2_reencoded_s0', 0, 'shortcut', .5), ('r2_reencoded_s1', 1, 'shortcut', .5),
          ('z113_decoy_s0', 0, 'decoy', .5), ('z113_decoy_s1', 1, 'decoy', .5),
          ('z113_clean_s0', 0, 'clean', .3), ('z113_clean_s1', 1, 'clean', .3), ('z113_cleanctrl_s2', 2, 'clean', .5)]


def p32():
    t0 = time.time(); o = jload(); res = {}
    for run, seed, mode, frac in FINALS:
        tk, z, y = get_probe(seed, mode, frac); ck = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1]
        rs = bz(load(ck, seed), tk, layers=True)
        res[run] = dict(ckpt=os.path.basename(ck), **{r['name']: dict(b_z=r['b_z'], b_z_rom=r['b_z_rom'], slot_share=r['slot_share']) for r in rs})
        print(f"[P3.2] {run:20s} " + '  '.join(f"{r['name']} {r['b_z']:.4f} (rom {r['b_z_rom']:.4f})" for r in rs), flush=True)
    o['P3_2'] = res; o['timing_p32_s'] = time.time() - t0; jsave(o)


def p34():
    t0 = time.time(); o = jload(); res = {}
    for run, seed, mode, frac in FINALS[:2]:
        tk, z, y = get_probe(seed, mode, frac); res[run] = {}
        for tag, ck in (('step500', f'runs/{run}/ckpt_000500.pt'), ('final', sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))[-1])):
            r = bz(load(ck, seed), tk, metric=True); G = r['metric'].numpy()
            w, V = np.linalg.eigh(G); w, V = w[::-1], V[:, ::-1]
            mass = (V[:, :10] ** 2).reshape(3, 128, 10).sum(1)          # (slot, eigvec)
            all_hint = (V ** 2).reshape(3, 128, -1).sum(1)[2]
            first = int(np.argmax(all_hint > 0.5)) + 1 if (all_hint > 0.5).any() else None
            res[run][tag] = dict(ckpt=os.path.basename(ck), top10_eigvals=w[:10].tolist(), top10_eig_share_of_trace=(w[:10] / w.sum()).tolist(),
                                 top10_hint_mass=mass[2].tolist(), top10_a_mass=mass[0].tolist(), top10_b_mass=mass[1].tolist(),
                                 rank_first_gt50_hint=first, hint_share_trace=float(np.trace(G[256:, 256:]) / np.trace(G)))
            print(f"[P3.4] {run} {tag}: hint share of trace {res[run][tag]['hint_share_trace']:.4f}; first >50%-hint eigvec rank {first}; top10 hint mass "
                  + ' '.join(f'{x:.2f}' for x in mass[2]), flush=True)
        rows = o['runs'][run]; e = [r['step'] for r in rows if r['b_z_rom'] > 0.10]
        res[run]['earliest_step_hint_share_gt_10pct'] = e[0] if e else None
        res[run]['hint_share_by_step_first8'] = [(r['step'], round(r['b_z_rom'], 4)) for r in rows[:8]]
    # the same "earliest" statistic on every run, as its null
    res['earliest_gt10pct_all_runs'] = {k: next((r['step'] for r in rows if r['b_z_rom'] > 0.10), None) for k, rows in o['runs'].items()}
    res['n_ckpt_gt10pct_all_runs'] = {k: f"{sum(r['b_z_rom'] > 0.10 for r in rows)}/{len(rows)}" for k, rows in o['runs'].items()}
    o['P3_4'] = res; o['timing_p34_s'] = time.time() - t0; jsave(o); print(json.dumps(res['earliest_gt10pct_all_runs']), json.dumps(res['n_ckpt_gt10pct_all_runs']))


def fig():
    from analysis import figstyle as fs
    import matplotlib.pyplot as plt
    fs.setup(); o = jload(); R = o['runs']; thr = o['threshold']['threshold']; thr_f = o['threshold']['threshold_floor']
    f, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for i, run in enumerate(('r2_reencoded_s0', 'r2_reencoded_s1')):
        a = ax[i]; rows = R[run]; s = np.array([r['step'] for r in rows])
        for r in rows:
            if r.get('h1_spurious') == 0 and r.get('reliance', 0) > 0.9:
                a.axvspan(r['step'] - 250, r['step'] + 250, facecolor='none', edgecolor=fs.INK2, hatch='////', linewidth=0, alpha=0.6)
        a.plot(s, [r['b_z'] for r in rows], color=fs.BLUE, ls='-', marker='o', ms=4, label='breaking score $b_z$')
        a.plot(s, [r['floor'] for r in rows], color=fs.ORANGE, ls='--', marker='s', ms=4, label='run-2 floor: NMI($\\hat y$, z)')
        a.axhline(thr, color=fs.BLUE, ls=':', lw=1.2); a.axhline(thr_f, color=fs.ORANGE, ls='-.', lw=1.2)
        a.annotate(f'$b_z$ threshold {thr:.3f}', (s[-1], thr), xytext=(-4, 3), textcoords='offset points', ha='right', fontsize=8, color=fs.INK)
        a.annotate(f'floor threshold {thr_f:.3f}', (s[-1], thr_f), xytext=(-4, 3), textcoords='offset points', ha='right', fontsize=8, color=fs.INK)
        a.set_title(f'Re-encoded model, seed {i}', fontsize=10); a.set_xlabel('training step'); a.set_ylim(-0.02, 1.02)
        if i == 0:
            from matplotlib.patches import Patch
            h, l = a.get_legend_handles_labels(); h.append(Patch(facecolor='none', edgecolor=fs.INK2, hatch='////', linewidth=0)); l.append('H$^1$ detector silent, reliance > 0.9')
            a.legend(h, l, loc='center right', bbox_to_anchor=(1.0, 0.36), fontsize=8); a.set_ylabel('statistic')
    a = ax[2]
    for run, c, ls, mk, lab in (('z113_shortcut_s0', fs.INK, '-', 'o', 'shortcut s0 (hint used)'), ('z113_decoy_s0', fs.MAGENTA, '--', '^', 'decoy s0 (hint independent of answer)'),
                                ('z113_clean_s0', fs.GREEN, ':', 'x', 'clean s0 (no hint in training)')):
        rows = R[run]; a.plot([r['step'] for r in rows], [r['b_z'] for r in rows], color=c, ls=ls, marker=mk, ms=3, markevery=3, lw=1.5, label=lab)
    a.axhline(thr, color=fs.BLUE, ls=':', lw=1.2); a.set_title('$b_z$: shortcut vs decoy vs clean', fontsize=10); a.set_xlabel('training step'); a.legend(fontsize=8); a.set_ylim(-0.02, 1.02)
    f.tight_layout(); os.makedirs('figures', exist_ok=True); f.savefig('figures/run3_bz_vs_floor.png', dpi=300); print('figures/run3_bz_vs_floor.png')


def score33():
    o = jload(); thr = o['threshold']['threshold']; res = {}
    for s in range(5):
        p = f'runs/r3_bz_s{s}/history.json'
        if not os.path.exists(p):
            res[f's{s}'] = dict(missing=True); continue
        H = json.load(open(p)); h = H['history']; by = {r['step']: r for r in h}
        late = [r for r in h if 6000 < r['step'] <= 12000]; fired = [r['step'] for r in h if r['fires']]
        res[f's{s}'] = dict(last_step=h[-1]['step'], first_fire=fired[0] if fired else None, lam_on_at=next((r['step'] for r in h if r.get('switched_on')), None),
                            dec_acc_6000=by.get(6000, {}).get('dec_acc'), test_acc_12000=by.get(12000, {}).get('test_acc'), dec_acc_12000=by.get(12000, {}).get('dec_acc'),
                            reliance_6000=by.get(6000, {}).get('reliance'), reliance_12000=by.get(12000, {}).get('reliance'),
                            n_late_checks=len(late), late_fires=[r['step'] for r in late if r['fires']], late_bz_max=max([r['b_z'] for r in late], default=None),
                            bz_trace=[(r['step'], round(r['b_z'], 4)) for r in h], wall_s=H['meta'].get('wall_s'))
    ok = [v for v in res.values() if not v.get('missing')]
    c1 = sum(1 for v in ok if v['dec_acc_6000'] is not None and v['dec_acc_6000'] > 0.98)
    c2 = sum(1 for v in ok if v['n_late_checks'] == 12 and not v['late_fires'])
    c3 = sum(1 for v in ok if v['test_acc_12000'] is not None and v['test_acc_12000'] >= 0.99)
    res['_verdict'] = dict(threshold=thr, dec_acc_gt_098_at_6000=f'{c1}/5', bz_below_thr_all_late_checks=f'{c2}/5', test_acc_ge_099_at_12000=f'{c3}/5',
                           P3_3_PASS=bool(c1 == 5 and c2 == 5 and c3 == 5))
    reached = min([v['last_step'] for v in ok], default=0)
    if len(ok) < 5 or reached < 12000:
        res['_verdict'].update(P3_3_PASS=None, status=f'NOT SCORED: training reached step {reached} of 12000 (counts above are over checks that exist)')
    o['P3_3'] = res; jsave(o); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', required=True); ap.add_argument('--runs', default=None); a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if a.only == 'p31' and a.runs:
        p31(a.runs.split(',')); sys.exit()
    dict(p31=p31, merge31=merge31, score31=score31, p32=p32, p34=p34, fig=fig, score33=score33)[a.only]()
