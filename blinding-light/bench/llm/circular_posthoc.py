"""POST HOC (after seeing prereg-phase-4 results): the adjacency score cannot separate a line from a circle (an optimal tour closes the endpoints).  Closure ratio at every layer:
 wrap = ||z_{n-1} - z_0|| / mean_i ||z_{i+1} - z_i||  (true order, PCA-6 space; circle ~ 1, line ~ n-1).  Reported for the layers where adjacency is perfect.  Never upgrades a verdict."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from bench.llm import circular as C
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def wrap_ratios(H, k=6):
    out = []
    for l in range(H.shape[0]):
        X = H[l] - H[l].mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); Z = U[:, :k] * S[:k]; n = len(Z)
        step = np.mean([np.linalg.norm(Z[i + 1] - Z[i]) for i in range(n - 1)]); out.append(float(np.linalg.norm(Z[n - 1] - Z[0]) / step))
    return out


res = {}
for name in ['gpt2', 'gpt2-medium', 'EleutherAI/pythia-70m', 'EleutherAI/pythia-410m', 'unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B-Instruct']:
    tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval(); res[name] = {}
    for s in ('days', 'months', 'numbers', 'colors', 'animals'):
        H = C.hidden_states(model, tok, C.SETS[s], C.TEMPLATES[s]); orders = C.blind_cycle_layers(H); sc = C.layer_scores(orders, len(C.SETS[s])); wr = wrap_ratios(H)
        good = [i for i in range(len(sc)) if sc[i] >= 0.85]
        res[name][s] = dict(layers_adj_ge_085=good, wrap_at_those=[wr[i] for i in good], min_wrap=(min(wr[i] for i in good) if good else None), wrap_all=wr)
    print(name, {s: (len(v['layers_adj_ge_085']), None if v['min_wrap'] is None else round(v['min_wrap'], 2)) for s, v in res[name].items()}, flush=True)
    del model; torch.cuda.empty_cache()
json.dump(res, open('results/phase4/circ_posthoc_wrap.json', 'w'))
