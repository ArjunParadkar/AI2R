import sys, json, torch, numpy as np; sys.path.insert(0, '.')
from bench.bio import gridrnn as G, rnn_analysis as A
for cond in ('conformal', 'noconf'):
    m = G.GridRNN(seed=900).cuda(); m.load_state_dict(torch.load(f'results/pilot_gridrnn_{cond}_900.pt')); m.eval()
    out, rm, idx = A.analyse(m)
    r = out.get('response'); print(cond, 'alive', out['n_alive'], 'modules', r and r['n_modules'], 'sil', r and round(r['silhouette'], 2), 'frac_gridlike', r and round(r['frac_gridlike'], 3), 'mean_r2', r and r['mean_r2_all'] and round(r['mean_r2_all'], 3))
    if r:
        for mm in r['modules']: print('   module', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mm.items()})
        print('   agreement', out['agreement']); print('   connectivity', out['connectivity']['k'], [(x['n'], x['b1']) for x in out['connectivity']['modules']]); print('   cross', out['cross_route'])
    np.save(f'results/pilot_ratemaps_{cond}.npy', rm[idx][:36])
