"""Phase 4B: formation of cyclic structure across Pythia training (one training run per model size; unit = that run).  Authors: Arjun Paradkar and Reva Paradkar.
Revisions step{N}.  Each revision is downloaded, scored (days, months, numbers control; max-over-layers adjacency + max-perm p) and deleted from the HF cache."""
import sys, json, os, shutil
sys.path.insert(0, '.')
from bench.llm import circular as C
from huggingface_hub import scan_cache_dir

STEPS = [0, 1, 4, 16, 64, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]


def purge(name, rev):
    try:
        info = scan_cache_dir(); h = [r.commit_hash for repo in info.repos if repo.repo_id == name for r in repo.revisions if rev in (r.refs or []) ]
        if h: info.delete_revisions(*h).execute()
    except Exception as e: print('purge failed', repr(e)[:100])


if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]; res = json.load(open(out)) if os.path.exists(out) else {}
    for st in STEPS:
        key = f'step{st}'
        if key in res: continue
        r = C.run_model(name, revision=key, sets=('days', 'months', 'numbers'))
        res[key] = {s: dict(best=v['best'], best_layer=v['best_layer'], p=v['p_maxperm'], layer_scores=v['layer_scores']) for s, v in r.items()}
        json.dump(res, open(out, 'w')); print(name, key, {s: (round(v['best'], 2), round(v['p'], 4)) for s, v in res[key].items()}, flush=True); purge(name, key)
