"""Score P6 and P7 on Z113 against spec 6.3, by the letter, from runs/*/sweep.json.

P6  'G jumps by > 0.4 within +-3 checkpoints of the test-accuracy jump; scrambled G < 0.2'
P7  'frequency birth times within +-3 checkpoints of grokking; birth order agrees across seeds (Kendall tau > 0.6)'
The grokking checkpoint is the one with the largest single-checkpoint rise in test accuracy.
Two readings of 'jump within +-3' are reported: endpoint difference G[g+3]-G[g-3] and range max-min.
G is the pre-registered statistic; G_interior is POST-HOC (endpoints excluded) and scored alongside.
"""
import sys, os, json
import numpy as np
from scipy.stats import kendalltau, spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.fig_grokking import births

CLEAN = ['z113_clean_s0', 'z113_clean_s1', 'z113_cleanctrl_s2']


def main():
    out = dict(P6={}, P7={})
    for run in CLEAN + ['z113_scrambled_s0', 'z113_shortcut_s0', 'z113_shortcut_s1']:
        p = f'runs/{run}/sweep.json'
        if not os.path.exists(p):
            continue
        R = json.load(open(p))['records']
        if 'G_interior' not in R[0]:
            print(run, 're-sweep pending'); continue
        te = np.array([r['test_acc'] or 0 for r in R]); g = int(np.argmax(np.diff(te))) + 1
        lo, hi = max(0, g - 3), min(len(R) - 1, g + 3)
        rec = dict(grok_ckpt=g, grok_step=R[g]['step'], test_rise=float(te[g] - te[g - 1]))
        for key in ('G', 'G_interior'):
            v = np.array([r[key] for r in R], float)
            rec[key] = dict(endpoint_jump=float(v[hi] - v[lo]), range_jump=float(np.nanmax(v[lo:hi + 1]) - np.nanmin(v[lo:hi + 1])),
                            final_mean_last10=float(np.nanmean(v[-10:])), before=float(v[lo]), after=float(v[hi]))
        # post-grok coupling of G to the loss (slingshots) -- needs history
        H = {h['step']: h for h in json.load(open(f'runs/{run}/history.json'))['history']}
        post = [r for r in R if r['step'] >= R[min(g + 4, len(R) - 1)]['step'] and r['step'] in H]
        if len(post) > 8 and 'train_loss' in H[post[0]['step']]:
            rec['postgrok_spearman_Gint_vs_trainloss'] = float(spearmanr([r['G_interior'] for r in post], [H[r['step']]['train_loss'] for r in post]).statistic)
        b = births(R); rec['births'] = {int(k): int(v) for k, v in b.items()}
        rec['births_within_3_of_grok'] = {int(k): bool(abs(v - g) <= 3) for k, v in b.items()}
        out['P6'][run] = rec
        print(f"{run}: grok ckpt {g} | G endpoint jump {rec['G']['endpoint_jump']:.3f} range {rec['G']['range_jump']:.3f} | "
              f"G_int endpoint {rec['G_interior']['endpoint_jump']:.3f} range {rec['G_interior']['range_jump']:.3f} | final G {rec['G']['final_mean_last10']:.3f} "
              f"G_int {rec['G_interior']['final_mean_last10']:.3f} | births {rec['births']} | Gint~loss {rec.get('postgrok_spearman_Gint_vs_trainloss')}")
    # P7 cross-seed order.  Seeds choose DIFFERENT key frequencies, so an order over frequency labels exists only
    # on the intersection.  Reported: size of the intersection, tau on it if >= 3, and the post-hoc alternative
    # (within a seed, does birth order follow final spectral power?).
    runs = [r for r in CLEAN if r in out['P6']]
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = out['P6'][runs[i]]['births'], out['P6'][runs[j]]['births']
            common = sorted(set(a) & set(b))
            tau = float(kendalltau([a[k] for k in common], [b[k] for k in common]).statistic) if len(common) >= 3 else None
            out['P7'][f'{runs[i]}|{runs[j]}'] = dict(freqs_a=sorted(a), freqs_b=sorted(b), common=common, kendall_tau=tau)
            print('P7 order', runs[i], runs[j], 'common freqs', common, 'tau', tau)
    for run in runs:
        R = json.load(open(f'runs/{run}/sweep.json'))['records']
        b = out['P6'][run]['births']
        power = {c['freq']: c['var'] for c in R[-1]['circles']}
        ks = [k for k in b if k in power]
        if len(ks) >= 3:
            out['P7'][f'{run}_posthoc_birth_vs_final_power_tau'] = float(kendalltau([b[k] for k in ks], [-power[k] for k in ks]).statistic)
    json.dump(out, open('results/p6_p7_score.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
