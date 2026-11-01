"""Per-input confidence baselines (CLAUDE.md section 6.4).  Every score: higher = more confident.

  max_softmax, neg_entropy, margin       from restricted-class logits (level A)
  ccgp, ccgp_per_input                   Bernardi et al. 2020 cross-condition generalisation (level C states)
  sharpness_per_input, weight_norm       Hessian trace of the per-input loss (level B)
"""
from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

ArrayOrDict = Union[np.ndarray, Dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# output-only baselines
# ---------------------------------------------------------------------------

def _logsoftmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _reduce(f, logits: ArrayOrDict, reduce: str):
    if not isinstance(logits, dict):
        return f(np.asarray(logits))
    v = np.stack([f(np.asarray(l)) for l in logits.values()], 0)
    if reduce == "mean":
        return v.mean(0)
    if reduce == "min":
        return v.min(0)
    if reduce == "sum":
        return v.sum(0)
    raise ValueError(reduce)


def max_softmax(logits: ArrayOrDict, reduce: str = "mean") -> np.ndarray:
    """max_c p(c|x).  A dict of behaviours is reduced across behaviours ('mean'|'min'|'sum')."""
    return _reduce(lambda z: np.exp(_logsoftmax(z)).max(-1), logits, reduce)


def neg_entropy(logits: ArrayOrDict, reduce: str = "mean") -> np.ndarray:
    """sum_c p log p (= -H, nats)."""
    def f(z):
        lp = _logsoftmax(z)
        return (np.exp(lp) * lp).sum(-1)
    return _reduce(f, logits, reduce)


def margin(logits: ArrayOrDict, reduce: str = "mean", space: str = "logit") -> np.ndarray:
    """top1 - top2, of the logits (space='logit') or of the probabilities (space='prob')."""
    def f(z):
        z = _logsoftmax(z) if space == "logit" else np.exp(_logsoftmax(z))
        s = np.sort(z, -1)
        return s[:, -1] - s[:, -2]
    return _reduce(f, logits, reduce)


# ---------------------------------------------------------------------------
# CCGP
# ---------------------------------------------------------------------------

def _svc(C=1.0, max_iter=5000):
    from sklearn.svm import LinearSVC
    return LinearSVC(C=C, max_iter=max_iter)


def _balanced_acc(y, yhat):
    accs = [np.mean(yhat[y == v] == v) for v in np.unique(y)]
    return float(np.mean(accs))


def _dichotomies(values, mode: str, rng, n_balanced: int):
    """Lists of value-sets forming the positive side."""
    values = list(values)
    if len(values) == 2:
        return [{values[1]}]
    if mode == "ovr":
        return [{v} for v in values]
    if mode == "balanced":
        half = len(values) // 2
        out, seen = [], set()
        for _ in range(50 * n_balanced):
            pos = frozenset(rng.choice(values, half, replace=False).tolist())
            key = min(pos, frozenset(values) - pos, key=lambda s: sorted(s))
            if key in seen:
                continue
            seen.add(key)
            out.append(set(pos))
            if len(out) == n_balanced:
                break
        return out
    raise ValueError(mode)


def _ccgp_one(H, y, cond, n_train_conditions, max_splits, mode, rng, C, n_balanced):
    """CCGP of one variable on states H (already standardised)."""
    vals = np.unique(y)
    if len(vals) < 2:
        return np.nan
    # a 'condition' for this variable is (condition id, value); conditions are normally pure
    key = np.array([f"{c}|{v}" for c, v in zip(cond, y)])
    ukeys, kidx = np.unique(key, return_inverse=True)
    kval = np.array([y[kidx == k][0] for k in range(len(ukeys))])
    scores = []
    for pos in _dichotomies(vals, mode, rng, n_balanced):
        side = np.isin(kval, list(pos)).astype(int)          # per condition: 1 = positive side
        P, Nn = np.where(side == 1)[0], np.where(side == 0)[0]
        if len(P) < 2 or len(Nn) < 2:
            continue
        nP = min(n_train_conditions or len(P) - 1, len(P) - 1)
        nN = min(n_train_conditions or len(Nn) - 1, len(Nn) - 1)
        combos_P = list(itertools.combinations(P, nP)) if _ncomb(len(P), nP) <= 4 * max_splits else None
        combos_N = list(itertools.combinations(Nn, nN)) if _ncomb(len(Nn), nN) <= 4 * max_splits else None
        total = (len(combos_P) if combos_P else np.inf) * (len(combos_N) if combos_N else np.inf)
        if total <= max_splits:
            splits = [(np.array(a), np.array(b)) for a in combos_P for b in combos_N]
        else:
            splits = [(rng.choice(P, nP, replace=False), rng.choice(Nn, nN, replace=False)) for _ in range(max_splits)]
        accs = []
        for trP, trN in splits:
            tr_k = np.concatenate([trP, trN])
            tr = np.isin(kidx, tr_k)
            te = ~tr
            yb = side[kidx]
            clf = _svc(C).fit(H[tr], yb[tr])
            accs.append(_balanced_acc(yb[te], clf.predict(H[te])))
        if accs:
            scores.append(np.mean(accs))
    return float(np.mean(scores)) if scores else np.nan


def _ncomb(n, k):
    from math import comb
    return comb(n, k)


def _standardise(Htr, *others):
    mu = Htr.mean(0)
    sd = Htr.std(0) + 1e-6
    return [(Htr - mu) / sd] + [(H - mu) / sd for H in others]


def ccgp(states: np.ndarray, var_labels: Union[np.ndarray, Dict[str, np.ndarray]], condition_ids: np.ndarray,
         n_train_conditions: Optional[int] = None, max_splits: int = 50, dichotomy: str = "ovr",
         n_balanced: int = 10, C: float = 1.0, seed: int = 0) -> Union[float, Dict[str, float]]:
    """Cross-condition generalisation performance (Bernardi et al. 2020, Cell 183:954).

    Trials are grouped by condition (`condition_ids`, e.g. the joint value of all task
    inputs).  For a dichotomy of a variable's values, a linear classifier (LinearSVC on
    standardised states) is trained on the trials of `n_train_conditions` conditions from each
    side (default: all but one condition per side, as in Bernardi et al.) and tested on the
    trials of the held-out conditions; CCGP is the mean test (balanced) accuracy over all
    choices of training-condition sets (randomly subsampled to `max_splits` when there are
    more), then over dichotomies.  Binary variables have one dichotomy; multi-valued ones use
    one-vs-rest dichotomies (dichotomy='ovr') or `n_balanced` random balanced splits of the
    values ('balanced').  Returns a scalar, or a dict per variable if a dict is given.
    Chance = 0.5.  Balanced accuracy is used because one-vs-rest sides are unbalanced.
    """
    rng = np.random.default_rng(seed)
    H = _standardise(np.asarray(states, dtype=np.float64))[0]
    cond = np.asarray(condition_ids)
    if not isinstance(var_labels, dict):
        return _ccgp_one(H, np.asarray(var_labels), cond, n_train_conditions, max_splits, dichotomy, rng, C, n_balanced)
    return {k: _ccgp_one(H, np.asarray(v), cond, n_train_conditions, max_splits, dichotomy, rng, C, n_balanced)
            for k, v in var_labels.items()}


def ccgp_per_input(states_by_site: Dict[str, np.ndarray], var_labels: Dict[str, np.ndarray],
                   condition_ids: np.ndarray, train_rows: np.ndarray, test_rows: np.ndarray,
                   n_folds: int = 5, site_select_splits: int = 10, C: float = 1.0, seed: int = 0,
                   sites: Optional[Sequence[str]] = None, return_details: bool = False):
    """Per-input CCGP score (our pre-registered adaptation; Bernardi et al. define CCGP only
    as a population-level accuracy).

    For each variable v:
      1. Site selection on TRAINING rows only: the site with the highest ccgp() over
         `train_rows` (all-but-one conditions per side, `site_select_splits` random splits).
      2. The unique conditions of the test rows are split into `n_folds` groups.  For each
         group F, one-vs-rest LinearSVCs (one per value u of v) are fit on the training rows
         whose condition is NOT in F (so no decoder has seen the held-out input's condition),
         on states standardised with those training rows.
      3. For a test input x in F with true value u, the score of v is the signed geometric
         margin of the u-vs-rest decoder, f_u(x) / ||w_u|| (positive = decoded correctly,
         larger = farther inside the u side).  If u never occurs in the fold's training rows
         the variable is skipped for x (nan).
    The per-input score is the mean over variables of these margins (nanmean).
    Returns scores (len(test_rows),) [, details: chosen site and per-variable margins].
    """
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(seed)
    train_rows, test_rows = np.asarray(train_rows), np.asarray(test_rows)
    cond = np.asarray(condition_ids)
    sites = list(states_by_site) if sites is None else list(sites)
    per_var = {}
    chosen = {}
    for name, lab in var_labels.items():
        lab = np.asarray(lab)
        best, best_s = -np.inf, None
        for s in sites:
            sc = ccgp(states_by_site[s][train_rows], lab[train_rows], cond[train_rows],
                      max_splits=site_select_splits, C=C, seed=int(rng.integers(1 << 31)))
            if np.isfinite(sc) and sc > best:
                best, best_s = sc, s
        chosen[name] = (best_s, best)
        m = np.full(len(test_rows), np.nan)
        if best_s is None:
            per_var[name] = m
            continue
        H = states_by_site[best_s]
        tconds = np.unique(cond[test_rows])
        perm = rng.permutation(tconds)
        folds = np.array_split(perm, min(n_folds, len(perm)))
        for F in folds:
            te_mask = np.isin(cond[test_rows], F)
            tr = train_rows[~np.isin(cond[train_rows], F)]
            if len(tr) == 0 or not te_mask.any():
                continue
            Htr, Hte = _standardise(H[tr].astype(np.float64), H[test_rows[te_mask]].astype(np.float64))
            ytr = lab[tr]
            yte = lab[test_rows[te_mask]]
            vals = np.unique(ytr)
            if len(vals) < 2:
                continue
            out = np.full(te_mask.sum(), np.nan)
            for u in vals:
                sel = yte == u
                if not sel.any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LinearSVC(C=C, max_iter=5000).fit(Htr, (ytr == u).astype(int))
                w = clf.coef_.ravel()
                out[sel] = (Hte[sel] @ w + clf.intercept_[0]) / (np.linalg.norm(w) + 1e-12)
            m[np.where(te_mask)[0]] = out
        per_var[name] = m
    M = np.stack(list(per_var.values()), 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(M, 0)
    if return_details:
        return score, {"sites": chosen, "margins": per_var}
    return score


# ---------------------------------------------------------------------------
# sharpness (Hessian trace) and weight norm
# ---------------------------------------------------------------------------

def _resolve(model, tokenizer, task):
    sysobj = None
    if hasattr(model, "model") and hasattr(model, "tok") and hasattr(model, "task"):
        sysobj = model
        model, tokenizer, task = sysobj.model, sysobj.tok, sysobj.task
    return sysobj, model, tokenizer, task


def _last_block_params(model, k_blocks):
    from .llm import _decoder_layers
    blocks = _decoder_layers(model)
    ps = []
    for blk in list(blocks)[-k_blocks:]:
        ps += [p for p in blk.parameters()]
    return ps


def _loss_inputs(task, tok, ex):
    ids, roles, readouts = task.render(ex, tok)
    if ex.get("class_ids") is not None:
        ci = ex["class_ids"]
        cls = {b: list(ci[b] if isinstance(ci, dict) else ci) for b in task.behaviours}
    elif hasattr(task, "example_class_ids"):
        c = list(task.example_class_ids(ex, tok))
        cls = {b: c for b in task.behaviours}
    else:
        c = list(task.class_token_ids(tok))
        cls = {b: c for b in task.behaviours}
    tg = task.targets([ex])
    pos = [readouts[b] for b in task.behaviours]
    tgt = [cls[b][int(tg[b][0])] for b in task.behaviours]
    return ids, pos, tgt


def per_input_loss(model, ids, pos, tgt):
    """Sum over behaviours of the full-vocabulary CE of the correct answer token."""
    import torch
    dev = next(model.parameters()).device
    out = model(input_ids=torch.as_tensor([ids], device=dev), use_cache=False)
    lg = out.logits[0, torch.as_tensor(pos, device=dev)]
    if lg.dtype in (torch.float16, torch.bfloat16):
        lg = lg.float()
    return torch.nn.functional.cross_entropy(lg, torch.as_tensor(tgt, device=dev), reduction="sum")


def sharpness_per_input(model, tokenizer=None, task=None, examples=None, k_blocks: int = 2, n_hutch: int = 8,
                        seed: int = 0, params=None, return_se: bool = False):
    """Hutchinson estimate of tr(∇²_θ L(x)) for each input x, where L(x) is the summed
    full-vocabulary cross-entropy of the correct answer token(s) at the task's readout
    positions (teacher forced), and θ = the parameters of the last `k_blocks` decoder blocks
    (or `params`).  tr H ≈ mean_i v_iᵀ H v_i with Rademacher v_i (n_hutch probes), Hv by double
    backprop.  NOTE: this is a sharpness score, so it is NOT sign-flipped; the caller decides
    the direction (flat = confident means use -sharpness).

    `model` may be an HFSystem (tokenizer and task taken from it) or a HF causal LM with
    `tokenizer` and `task` given.  Runs under the SDPA MATH backend (the efficient-attention
    backward kernel is wrong on the RTX 5070 / torch 2.11); prefer a float32 model, bf16
    Hessians are noisy.  Returns (N,) traces [, (N,) standard errors].
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _, model, tokenizer, task = _resolve(model, tokenizer, task)
    P = list(params) if params is not None else _last_block_params(model, k_blocks)
    pid = {id(p) for p in P}
    flags = [(p, p.requires_grad) for p in model.parameters()]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev)
    tr_out, se_out = [], []
    try:
        for p, _ in flags:
            p.requires_grad_(id(p) in pid)
        with sdpa_kernel(SDPBackend.MATH):
            for n, ex in enumerate(examples):
                ids, pos, tgt = _loss_inputs(task, tokenizer, ex)
                loss = per_input_loss(model, ids, pos, tgt)
                grads = torch.autograd.grad(loss, P, create_graph=True)
                g.manual_seed(seed * 1000003 + n)
                est = []
                for i in range(n_hutch):
                    vs = [(torch.randint(0, 2, p.shape, generator=g, device=dev) * 2 - 1).to(p.dtype) for p in P]
                    gv = sum((gr * v).sum() for gr, v in zip(grads, vs))
                    Hv = torch.autograd.grad(gv, P, retain_graph=i < n_hutch - 1)
                    est.append(float(sum((h.float() * v.float()).sum() for h, v in zip(Hv, vs))))
                est = np.array(est)
                tr_out.append(est.mean())
                se_out.append(est.std(ddof=1) / np.sqrt(len(est)) if len(est) > 1 else np.nan)
                del grads, loss
    finally:
        for p, f in flags:
            p.requires_grad_(f)
        model.train(was_training)
    if return_se:
        return np.array(tr_out), np.array(se_out)
    return np.array(tr_out)


def weight_norm(model, k_blocks: Optional[int] = None, params=None) -> float:
    """L2 norm of all parameters (or of the last k_blocks decoder blocks / `params`)."""
    import torch
    _, model, _, _ = _resolve(model, None, None)
    if params is None:
        params = _last_block_params(model, k_blocks) if k_blocks else list(model.parameters())
    with torch.no_grad():
        return float(torch.sqrt(sum((p.float() ** 2).sum() for p in params)).item())
