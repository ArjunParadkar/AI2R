import sys, time, json, torch; sys.path.insert(0, '.')
from bench.bio import gridrnn as G, rnn_analysis as A
cond, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); t = time.time()
m, hist, ck = G.train(cond, seed, steps=steps, ckpt_every=max(steps // 8, 100), log=print)
print(cond, 'time', round(time.time() - t), json.dumps(hist[-2:]), flush=True)
torch.save(ck[max(ck)], f'results/pilot_gridrnn_{cond}_{seed}.pt')
for step in sorted(ck)[1:]:
    mm = G.GridRNN(seed=seed).cuda(); mm.load_state_dict(ck[step]); mm.eval(); out, rm, idx = A.analyse(mm); r = out.get('response')
    print('STEP', step, 'alive', out['n_alive'], 'modules', r and r['n_modules'], 'frac_gridlike', r and round(r['frac_gridlike'], 3), 'mean_r2', r and r['mean_r2_all'] and round(r['mean_r2_all'], 3), flush=True)
