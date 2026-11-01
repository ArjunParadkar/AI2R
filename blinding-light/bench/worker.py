"""Resumable queue worker.  python bench/worker.py grid.jsonl SHARD NSHARDS [outdir]
Each grid line is a cfg dict {id, task, arch, regime, seed, hp[, extra]}.  One JSON per cfg in outdir (rows + trajectory), errors in <id>.err.
Authors: Arjun Paradkar and Reva Paradkar."""
import sys, os, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
from whiteboxtopo.numerics import repro
from bench import zoo

grid, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else 'results/phase2'
os.makedirs(out, exist_ok=True); os.environ.setdefault('WB_SAVE_DIR', out + '_models')
cfgs = [json.loads(l) for l in open(grid) if l.strip()]
mine = [c for i, c in enumerate(cfgs) if i % nsh == shard]
for c in mine:
    p = f"{out}/{c['id']}.json"
    if os.path.exists(p): continue
    try:
        repro.set_determinism(c['seed']); os.environ['WB_CFG_ID'] = c['id']
        t0 = time.time()
        for attempt in range(4):                      # CUDA OOM from sibling workers is transient: back off and retry, never record it as a result
            try: rows = zoo.run_config({k: v for k, v in c.items() if k != 'id'}, progress=lambda s: None); break
            except Exception as e:
                import torch
                if 'out of memory' in str(e).lower() and attempt < 3: torch.cuda.empty_cache(); time.sleep(45 * (attempt + 1)); continue
                raise
        for r in rows:
            r['cfg_id'] = c['id']
            for k, v in list(r.items()):
                if hasattr(v, 'tolist'): r[k] = v.tolist()
        json.dump(rows, open(p + '.tmp', 'w'), default=float); os.replace(p + '.tmp', p)
        print(c['id'], 'ok', round(time.time() - t0), 's', flush=True)
    except Exception as e:
        open(f"{out}/{c['id']}.err", 'w').write(traceback.format_exc()); print(c['id'], 'ERR', repr(e)[:200], flush=True)
