import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, '.')
from bench import zoo, clockpizza as CP, stats as S
rows = []
for f in sorted(glob.glob('results/pilot3/*.json')):
    r = json.load(open(f)); fin = [x for x in r if x['snapshot'] == 'final'][0]; cid = os.path.basename(f)[:-5]
    pt = f'results/pilot3_models/{cid}_final.pt'
    if not os.path.exists(pt): continue
    task = zoo.get_task(fin['task']); al = json.loads(fin['hp'])['model']['alpha']
    m = zoo.make_model(task, 'tf1', fin['seed'], alpha=al); m.load_state_dict(torch.load(pt)); m.eval()
    d = CP.all_descriptors(m, task.n); d.update(task=fin['task'], alpha=al, seed=fin['seed'], test_acc=fin['test_acc'], d1_eps=fin['d1_eps'], d1_ampcv=fin['d1_ampcv'], kappa=fin['kappa'],
                                               **{k: fin[k] for k in ('base_entropy', 'base_wnorm', 'base_margin', 'base_hess_trace', 'fourier_top5')}); rows.append(d); print(cid, flush=True)
json.dump(rows, open('results/clockpizza_pilot.json', 'w'), default=float)
for t in ('mod59', 'mod113'):
    for name in ('spec_marginal_frac', 'spec_marginal_frac_key', 'spec_joint', 'spec_marg', 'amp_cv', 'margin_cv', 'swap_asym', 'd1_eps', 'd1_ampcv', 'base_entropy', 'base_wnorm', 'base_margin', 'base_hess_trace', 'fourier_top5'):
        c = [r for r in rows if r['task'] == t and r['test_acc'] >= 0.99]; pz = [r[name] for r in c if r['alpha'] == 0.0]; ck = [r[name] for r in c if r['alpha'] == 1.0]
        if len(pz) >= 3 and len(ck) >= 3: print(t, name.ljust(24), 'n', len(pz), len(ck), 'AUROC(pizza>clock)', round(S.auroc(pz, ck), 3), 'mean', round(np.mean(pz), 3), round(np.mean(ck), 3))
