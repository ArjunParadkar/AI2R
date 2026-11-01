"""Phase 5 scoring: reads the sealed key ONLY after predictions are committed and tagged.  python bench/blind/score.py <key path>"""
import sys, json, glob, hashlib
import numpy as np
sys.path.insert(0, '.')
from bench import stats as S
from scipy.stats import spearmanr
pred = json.load(open('results/blind/predictions.json'))['predictions']; kp = sys.argv[1]; blob = open(kp, 'rb').read(); com = json.load(open('results/blind/key_commitment.json'))
assert hashlib.sha256(blob).hexdigest() == com['sha256'], 'key does not match the committed hash'; key = json.loads(blob)
fin = {k: v for k, v in pred.items() if not k.endswith('_midsnap')}; gen = np.array([key[k]['test_acc'] >= 0.9 for k in fin]); res = {}
res['n_models'] = len(fin); res['frac_generalising_true'] = float(gen.mean())
res['generalises_auroc'] = S.boot_ci_auroc([fin[k]['p_generalises'] for k in fin if key[k]['test_acc'] >= 0.9], [fin[k]['p_generalises'] for k in fin if key[k]['test_acc'] < 0.9])
res['test_acc_mae'] = float(np.mean([abs(fin[k]['pred_test_acc'] - key[k]['test_acc']) for k in fin])); res['test_acc_spearman'] = float(spearmanr([fin[k]['pred_test_acc'] for k in fin], [key[k]['test_acc'] for k in fin]).statistic)
res['baseline_mean_pred_mae'] = float(np.mean([abs(np.mean([key[j]['test_acc'] for j in fin]) - key[k]['test_acc']) for k in fin]))      # constant predictor
mech = [(k, key[k]['alpha']) for k in fin if key[k]['regime'] == 'clean' and key[k]['test_acc'] >= 0.99]
res['mechanism_accuracy'] = float(np.mean([(fin[k]['mechanism'] == 'clock') == (a == 1.0) for k, a in mech])) if mech else None; res['mechanism_n'] = len(mech)
au = []
for k in fin:
    err = (np.array(fin[k]['model_pred_te']) != np.array(key[k]['Yte'])); sc = np.array(fin[k]['per_input_error_score'])
    if 0 < err.sum() < len(err): au.append(S.auroc(sc[err], sc[~err]))
res['failure_input_auroc_per_model'] = dict(mean=float(np.mean(au)) if au else None, n=len(au))
mid = {k: v for k, v in pred.items() if k.endswith('_midsnap')}; res['midsnap_n'] = len(mid)
json.dump(res, open('results/blind/score.json', 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
