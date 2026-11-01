"""Phase 4: blind recovery of cyclic structure (days Z_7, months Z_12) in language-model representations.  Authors: Arjun Paradkar and Reva Paradkar.
Access level C (hidden states).  The cycle is NEVER supplied: PCA -> optimal Hamiltonian cycle (Held-Karp) on the item representations -> compare the
recovered cyclic order with the true one only at scoring time.  Layer selection is controlled by a max-over-layers permutation null.
Controls: number words one..seven (ordered, NOT cyclic), colours, animals (no order); each is scored against an ARBITRARY fixed 'true' cyclic order."""
import itertools, json, sys, os, math
import numpy as np

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
NUMS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black']
ANIMALS = ['dog', 'cat', 'horse', 'cow', 'sheep', 'goat', 'pig']
SETS = dict(days=DAYS, months=MONTHS, numbers=NUMS, colors=COLORS, animals=ANIMALS)
TEMPLATES = {
    'days': ['Today is {x}.', 'The meeting is on {x}.', 'See you next {x}.', 'On {x} I usually stay home.', 'She was born on a {x}.', 'The store is closed on {x}.', 'Every {x} we go running.', 'It happened last {x}.'],
    'months': ['It was in {x}.', 'The month of {x} is cold.', 'My birthday is in {x}.', 'We moved here in {x}.', 'Next {x} we travel.', 'The festival takes place in {x}.', 'Prices rose in {x}.', 'Born in {x}, she grew up here.'],
    'numbers': ['I counted {x}.', 'There are {x} of them.', 'He has {x} books.', 'She waited {x} minutes.', 'We need {x} chairs.', 'Only {x} remained.', 'Take {x} steps.', 'It cost {x} dollars.'],
    'colors': ['The car is {x}.', 'She wore a {x} dress.', 'The flag is {x}.', 'I painted it {x}.', 'A {x} door.', 'The sky turned {x}.', 'He chose {x}.', 'It looked {x}.'],
    'animals': ['I saw a {x}.', 'The {x} ran away.', 'She owns a {x}.', 'A {x} lives here.', 'He fed the {x}.', 'The {x} was loud.', 'We found a {x}.', 'That is a {x}.'],
}


def hidden_states(model, tok, words, templates, device='cuda'):
    """-> (L+1, n_words, d): residual stream at the last sub-token of the word, averaged over templates."""
    import torch
    out = None
    for w_i, w in enumerate(words):
        acc = []
        for t in templates:
            pre, post = t.split('{x}'); ids_pre = tok(pre, add_special_tokens=False)['input_ids'] if pre else []
            ids_w = tok((' ' if pre and not pre.endswith(' ') else '') + w if pre else w, add_special_tokens=False)['input_ids']
            ids = ids_pre + ids_w; pos = len(ids) - 1
            if tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False): ids = [tok.bos_token_id] + ids; pos += 1
            with torch.no_grad(): hs = model(torch.tensor([ids], device=device), output_hidden_states=True).hidden_states
            acc.append(torch.stack([h[0, pos].float() for h in hs]).cpu().numpy())
        a = np.mean(acc, 0)
        if out is None: out = np.zeros((a.shape[0], len(words), a.shape[1]))
        out[:, w_i] = a
    return out


def held_karp_cycle(D):
    n = len(D); INF = 1e18; full = 1 << (n - 1); dp = np.full((full, n - 1), INF); par = np.full((full, n - 1), -1, int)
    for j in range(n - 1): dp[1 << j, j] = D[n - 1, j]
    for mask in range(1, full):
        for j in range(n - 1):
            if not (mask >> j) & 1 or dp[mask, j] >= INF: continue
            base = dp[mask, j]
            for k in range(n - 1):
                if (mask >> k) & 1: continue
                nm = mask | (1 << k); v = base + D[j, k]
                if v < dp[nm, k]: dp[nm, k] = v; par[nm, k] = j
    last = int(np.argmin([dp[full - 1, j] + D[j, n - 1] for j in range(n - 1)])); order = [n - 1]; mask = full - 1; j = last
    while j != -1: order.append(j); pj = par[mask, j]; mask ^= (1 << j); j = pj
    return order


def adjacency_score(order, n):
    """fraction of the n true cyclic neighbour pairs (i, i+1 mod n) that are adjacent in the recovered cycle."""
    pos = {x: i for i, x in enumerate(order)}; hit = 0
    for i in range(n):
        d = abs(pos[i] - pos[(i + 1) % n]); hit += (d == 1 or d == n - 1)
    return hit / n


def blind_cycle_layers(H, k_pca=6):
    """H: (L+1, n, d).  Per layer: centre, PCA(k), distance matrix, optimal cycle.  Returns list of orders."""
    orders = []
    for l in range(H.shape[0]):
        X = H[l] - H[l].mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); Z = U[:, :k_pca] * S[:k_pca]
        D = np.linalg.norm(Z[:, None] - Z[None], axis=2); orders.append(held_karp_cycle(D))
    return orders


def layer_scores(orders, n): return np.array([adjacency_score(o, n) for o in orders])


def max_perm_pvalue(orders, n, observed_max, n_perm=20000, seed=0):
    """P(max over layers of adjacency score >= observed) when the true cyclic order is a uniformly random relabelling (selection-controlled)."""
    rng = np.random.default_rng(seed); cnt = 0; pos = [{x: i for i, x in enumerate(o)} for o in orders]
    for _ in range(n_perm):
        pi = rng.permutation(n); best = 0
        for p in pos:
            hit = sum((abs(p[pi[i]] - p[pi[(i + 1) % n]]) in (1, n - 1)) for i in range(n)) / n; best = max(best, hit)
            if best >= observed_max: break
        cnt += best >= observed_max
    return (cnt + 1) / (n_perm + 1)


def run_model(name, revision=None, device='cuda', sets=('days', 'months', 'numbers', 'colors', 'animals'), dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, revision=revision); model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype or torch.float16).to(device).eval()
    res = {}
    for s in sets:
        words = SETS[s]; H = hidden_states(model, tok, words, TEMPLATES[s], device); orders = blind_cycle_layers(H); sc = layer_scores(orders, len(words))
        res[s] = dict(layer_scores=sc.tolist(), best_layer=int(sc.argmax()), best=float(sc.max()), p_maxperm=max_perm_pvalue(orders, len(words), float(sc.max())), orders=[list(map(int, o)) for o in orders])
    del model; torch.cuda.empty_cache(); return res


if __name__ == '__main__':
    name = sys.argv[1]; out = sys.argv[2]; rev = sys.argv[3] if len(sys.argv) > 3 else None
    r = run_model(name, rev); json.dump(r, open(out, 'w'))
    print(name, rev, {s: (round(v['best'], 2), v['best_layer'], round(v['p_maxperm'], 4)) for s, v in r.items()})
