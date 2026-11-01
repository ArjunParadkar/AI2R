"""Summarise the D3/D4 interrupt arms from runs/r2_align_*/history.json (works on partial histories)."""
import json, glob, os, re
import numpy as np
out = {}
for arm, label in (('R', 'pre-registered penalty, schedule R (x10, <=3 rounds) -- POST-HOC trigger'), ('G', 'pre-registered penalty, schedule G (x3, <=5 rounds) -- POST-HOC trigger'),
                   ('Rrun1pen', "run-1 penalty, schedule R -- POST-HOC arm A-4")):
    rows = []
    for s in range(5):
        p = f'runs/r2_align_{arm}_s{s}/history.json'
        if not os.path.exists(p):
            continue
        H = json.load(open(p))['history']; by = {h['step']: h for h in H}; last = H[-1]
        at6 = by.get(6000); ext = [h for h in H if h['step'] > 6000]
        rows.append(dict(seed=s, last_step=last['step'], first_fire=next((h['step'] for h in H if h['fires']), None), max_lam=max(h['lam'] for h in H),
                         dec_acc_6000=None if at6 is None else at6['dec_acc'], reliance_6000=None if at6 is None else at6['reliance'],
                         d3_met=None if at6 is None else bool(at6['dec_acc'] > 0.98),
                         ext_checks=len(ext), ext_firings=int(sum(h['fires'] for h in ext)), ext_min_dec=min((h['dec_acc'] for h in ext), default=None),
                         dec_acc_last=last['dec_acc'], reliance_last=last['reliance']))
    if rows:
        done6 = [r for r in rows if r['d3_met'] is not None]
        out[arm] = dict(label=label, seeds=rows, n_d3_met=int(sum(r['d3_met'] for r in done6)), n_scored=len(done6),
                        d4_holds=[r['seed'] for r in rows if r['d3_met'] and r['ext_checks'] >= 12 and r['ext_firings'] == 0])
        print(arm, f"D3 met {out[arm]['n_d3_met']}/{len(done6)} |", ' '.join(f"s{r['seed']}:{r['dec_acc_6000'] if r['dec_acc_6000'] is None else round(r['dec_acc_6000'],3)}(lam{r['max_lam']:g},last{r['last_step']})" for r in rows), '| D4 holds', out[arm]['d4_holds'])
json.dump(out, open('results/run2/d34_summary.json', 'w'), indent=1)
