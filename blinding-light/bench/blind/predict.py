"""Phase 5 (internal blind): predictions from PUBLIC material only (weights, task name, arch, seed, training inputs+labels, held-out inputs without labels).
Fit on Phase-2 rows (results/phase2, modular tf1 models, seeds 1-10) — the blind seeds 7001+ never enter.  Writes results/blind/predictions.json.  The sealed key is NOT read.
Targets: (1) P(generalises: held-out acc >= 0.9)  (2) predicted held-out accuracy  (3) mechanism class clock(alpha=1)/pizza(alpha=0) via swap_asym threshold from Phase 2
(4) per-held-out-input error scores (1 - kappa_x)  (5) formation checkpoint: NOT predicted (no trajectories are released)."""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, '.')
from bench import zoo
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

PUB = 'results/blind/public'; FEATS = ['kappa', 'q_logit', 'fiber_tight', 'fourier_top5', 'cover_r0.5', 'base_wnorm', 'base_entropy', 'swap_asym']


def phase2_table():
    rows = []
    for f in glob.glob('results/phase2/*.json'):
        for r in json.load(open(f)):
            if r['task'] in ('mod59', 'mod97', 'mod113') and r['arch'] == 'tf1' and r['regime'] in ('clean', 'scrambled', 'nograk', 'hybrid', 'midgrok') and all(r.get(k) is not None for k in FEATS): rows.append(r)
    return rows


def features(model, task, data, model0):
    row, per_input, st = zoo.instruments(task, 'tf1', model, model0, data, 'blind', 0, {}, 'final'); return row, st


def main():
    rows = phase2_table(); assert len(rows) > 100, 'Phase 2 data missing'
    X = np.array([[r[k] for k in FEATS] for r in rows]); acc = np.array([r['test_acc'] for r in rows]); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z, acc >= 0.9); reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(Z, acc)
    cp = [r['swap_asym'] for r in rows if r['regime'] == 'clean' and r['test_acc'] >= 0.99 and json.loads(r['hp']).get('model', {}).get('alpha', 1.0) == 1.0]
    pz = [r['swap_asym'] for r in rows if r['regime'] == 'clean' and r['test_acc'] >= 0.99 and json.loads(r['hp']).get('model', {}).get('alpha', 1.0) == 0.0]; thr = float((np.median(cp) + np.median(pz)) / 2)
    preds = {}
    for f in sorted(glob.glob(f'{PUB}/m*.meta.json')):
        meta = json.load(open(f)); mid = meta['id']
        for snap in ('', '_midsnap'):
            pt = f'{PUB}/{mid}{snap}.pt'
            if not os.path.exists(pt): continue
            task = zoo.get_task(meta['task']); d = np.load(f'{PUB}/{mid}_data.npz'); data = zoo.build_data(task, meta['seed'], 'clean')          # probe inputs only come from the public task definition
            data.update(Xtr=d['Xtr'], Ytr=d['Ytr'], Xte=d['Xte'])
            m = zoo.make_model(task, 'tf1', meta['seed']); m0 = zoo.make_model(task, 'tf1', meta['seed']); sdict = torch.load(pt)
            m.load_state_dict(sdict); m.eval(); row, st = features(m, task, data, m0); z = (np.array([[row[k] for k in FEATS]]) - mu) / sd
            Xt = torch.as_tensor(d['Xte'], device=zoo.DEV); stx = zoo.orbit_stats(m, Xt, task.group()); lg = zoo.predict(m, Xt).argmax(-1).cpu().numpy().ravel()
            preds[mid + snap] = dict(p_generalises=float(clf.predict_proba(z)[0, 1]), pred_test_acc=float(reg.predict(z)[0]), swap_asym=float(row['swap_asym']), mechanism=('clock' if row['swap_asym'] > thr else 'pizza'),
                                      per_input_error_score=(1 - stx['kappa']).tolist(), model_pred_te=lg.tolist(), features={k: float(row[k]) for k in FEATS})
            print(mid + snap, round(preds[mid + snap]['p_generalises'], 3), round(preds[mid + snap]['pred_test_acc'], 3), preds[mid + snap]['mechanism'], flush=True)
    json.dump(dict(swap_threshold=thr, n_phase2_rows=len(rows), predictions=preds), open('results/blind/predictions.json', 'w'))


if __name__ == '__main__': main()
