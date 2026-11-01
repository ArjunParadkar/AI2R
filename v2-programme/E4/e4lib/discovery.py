"""Fixed circuit-discovery pipeline (SPEC 'Fixed discovery pipeline'):
node attribution patching (mean corruption) + greedy pruning to a minimal faithful circuit at F >= 0.9,
two passes, re-scoring every 16 accepted removals. Cost = number of ablated forward passes."""
import time, numpy as np, torch
from .tasks import metric

@torch.no_grad()
def _m(model, tokens, targets, keep, means):
    return metric(model(tokens, keep=keep, means=means), targets).item()

def faithfulness(model, tokens, targets, keep, means, m_clean, m_corr):
    if abs(m_clean - m_corr) < 1e-8: return float("nan")
    return (_m(model, tokens, targets, keep, means) - m_corr) / (m_clean - m_corr)

def discover_circuit(model, tokens, targets, means, seed=0, F_target=0.9, rescore_every=16, n_passes=2):
    """Returns dict(keep, size, frac, F, cost, wall). keep: bool tensor over nodes."""
    model.eval(); t0 = time.time(); N = model.n_nodes
    rng = np.random.default_rng(seed)
    full = torch.ones(N, dtype=torch.bool); none = torch.zeros(N, dtype=torch.bool)
    m_clean = _m(model, tokens, targets, full, means); m_corr = _m(model, tokens, targets, none, means)
    gap = m_clean - m_corr; cost = 2
    keep = full.clone()
    def F(k):
        nonlocal cost; cost += 1
        return (_m(model, tokens, targets, k, means) - m_corr) / gap if abs(gap) > 1e-8 else float("nan")
    if not np.isfinite(F(full)) :  # degenerate gap
        return dict(keep=keep, size=N, frac=1.0, F=float("nan"), cost=cost, wall=time.time() - t0, gap=gap)
    for _ in range(n_passes):
        sc = model.node_scores(tokens, targets, means).abs().numpy()
        sc = sc + rng.normal(0, 1e-9 * (sc.max() + 1e-12), N)          # seed-dependent tie-break
        order = [i for i in np.argsort(sc) if keep[i]]; accepted = 0; pos = 0
        while pos < len(order):
            i = order[pos]; pos += 1
            if not keep[i]: continue
            keep[i] = False
            if F(keep) >= F_target:
                accepted += 1
                if accepted % rescore_every == 0:                         # re-rank survivors
                    sc = model.node_scores(tokens, targets, means * 1)  # scores on the full model input;
                    # ablation-aware rescoring: score with the current circuit as the model
                    sc = _scores_under(model, tokens, targets, means, keep).abs().numpy()
                    sc = sc + rng.normal(0, 1e-9 * (sc.max() + 1e-12), N)
                    order = [j for j in np.argsort(sc) if keep[j] and j not in set(order[:pos])]
                    pos = 0
            else:
                keep[i] = True
    Ff = F(keep)
    return dict(keep=keep, size=int(keep.sum()), frac=float(keep.sum() / N), F=float(Ff), cost=cost,
                wall=time.time() - t0, gap=gap)

def _scores_under(model, tokens, targets, means, keep):
    """Node attribution scores with the currently ablated nodes held at their means."""
    model.zero_grad(set_to_none=True)
    logits = model(tokens, keep=keep, means=means, record=True)
    m = metric(logits, targets); outs = model._node_out
    grads = torch.autograd.grad(m, outs, allow_unused=True)
    return torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item() for o, g, mu in zip(outs, grads, means)])

def jaccard(a, b):
    a, b = set(np.flatnonzero(np.asarray(a))), set(np.flatnonzero(np.asarray(b)))
    return len(a & b) / max(1, len(a | b))
