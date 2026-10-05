"""Fill the P3.2 placeholders (report, CLAIMS.md, FINDINGS4_P3.md) from the current interrupt histories; safe to re-run."""
import json, os, subprocess, sys
subprocess.run([sys.executable, 'analysis/run4_score.py'], capture_output=True)
p = json.load(open('results/run4/score.json'))['P3']['P3_2']; v = p['seeds']; ks = sorted(v, key=int)
if p['scored']:
    nums = ' / '.join(f"{v[k]['dec6000']:.3f}" for k in ks); fires = sum(v[k]['ext_fires'] for k in ks); tests = ' / '.join(f"{v[k]['test12000']:.3f}" for k in ks)
    short = f"decorrelated accuracy at step 6000: {nums}; floor re-firings in (6000, 12000]: {fires}; in-distribution test at 12000: {tests}"
    verdict = 'PASS' if p['PASS'] else 'FAIL'; tag = f'<span class="tag {"pass" if p["PASS"] else "fail"}">{"✓ PASS" if p["PASS"] else "✗ FAIL"}</span>'
    text = (f"<strong>The interrupt, finally scored.</strong> Functional-floor trigger (it fired at step 500 on every seed), run 1's presence-and-value KL penalty, λ = 3 switched on once and never escalated, five seeds: {short}. "
            + ("All three registered clauses hold on five of five seeds. It is not topological and it needs the nuisance to be known." if p['PASS'] else "By the letter this fails; the numbers are as given."))
else:
    prog = ', '.join(f"seed {k}: step {v[k]['last']}" + (f", {v[k]['dec6000']:.3f} at 6k" if v[k].get('dec6000') is not None else '') for k in ks)
    short = f"still running on the power-capped GPU ({prog}); not yet scored"; verdict = 'not yet scored'; tag = '<span class="tag expl">— running</span>'
    text = f"<strong>The interrupt</strong> (functional-floor trigger, run 1's KL penalty, λ = 3 once, five seeds, 12,000 steps) is still running on the power-capped GPU: {prog}. It is reported as not yet scored, not as a result."
t = open('report/run4.template.html').read().replace('__P32__', short).replace('__P32TAG__', tag).replace('__P32TEXT__', text); open('report/blinding-light-run4.html', 'w').write(t)
for f in ('CLAIMS.md', 'FINDINGS4_P3.md'):
    src = f + '.p32src'
    if not os.path.exists(src):
        open(src, 'w').write(open(f).read())
    open(f, 'w').write(open(src).read().replace('__P32__', short).replace('__P32V__', verdict))
print(verdict, '|', short)
