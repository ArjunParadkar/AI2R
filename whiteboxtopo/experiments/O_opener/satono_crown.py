#!/usr/bin/env python3
"""O2.1 Satono Crown — open() on GPT-2 small (FINAL_SPEC §3 steps 1-9, §5 T2, §6, §7, §8.2 O2.1) for the three
T2 behaviours: IOI, induction, greater-than. Pre-registration: ``SPEC_O2.md`` (written before any O2 compute).

  python satono_crown.py --tasks ioi induction greater_than            # full run (resumable, per task and per cell)
  python satono_crown.py --smoke                                        # tiny end-to-end check of all three
  python satono_crown.py --tasks ioi --cells P0 P1 --force              # re-run selected cells into the existing JSON
  python satono_crown.py --audit                                        # O2.2 Narita Top Road: carry the F11 audit rows

Outputs (results/open/): ``satono_crown_<task>.json`` per task (every cell with status / value / criterion / null /
provenance / wall), ``satono_crown_<task>.md`` (the whiteboxtopo.open Scorecard), the merged
``satono_crown_gpt2.json`` with "cells" + "scorecard" for the O5 grid assembler, ``narita_top_road.json`` for O2.2,
and one row per cell in results/O2_cells.csv.

Environment: the TransformerLens venv (``~/Whitebox/v2-programme/.venv-tl/bin/python``) has torch + transformers but
NOT an editable install of whiteboxtopo, so ROOT is prepended to sys.path below. The weights are held as a Hugging
Face ``GPT2LMHeadModel`` because that is the class ``whiteboxtopo.symmetry.gpt2`` renames (SPEC_O2 §0); a
TransformerLens ``HookedTransformer`` has no gauge backend.

Cells whose instrument does not apply, whose module fails, or which exceed their pre-registered budget are recorded
"not_reached: <reason>" — never skipped silently (Operating Constitution §12.4: negatives are deliverables).
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RES = os.path.join(HERE, "results"); OUT = os.path.join(RES, "open")
os.makedirs(OUT, exist_ok=True)
CELLS_CSV = os.path.join(RES, "O2_cells.csv")
CELL_FIELDS = ["task", "cell", "status", "value", "criterion", "provenance", "wall"]

LAYER = 6                      # mid layer: the residual stream after block 6 — the layer F11 audited (SPEC_O2 §0)
TASKS = ("ioi", "induction", "greater_than")
CELLS = ["gauge", "P0", "P1", "P2", "glue", "R3", "R4", "R5", "R6", "P6"]
ART_CV = 1e-3                  # SPEC_O K-O-2 artifact bar
ATTR_CV = 1e-4                 # fp32 realisation of "exact" for attributions (SPEC_O2 §5)
F_TARGET = 0.9
COVER_K = 10; GLUE_TAU = 0.5; GLUE_KE = 32
# pre-registered sizes (SPEC_O2 §1.4): (n_discovery, n_heldout) per task; --n-prompts overrides both
NPROMPTS = {"ioi": (128, 128), "induction": (64, 64), "greater_than": (128, 128)}
SMOKE_N = {"ioi": (16, 16), "induction": (8, 8), "greater_than": (16, 16)}
BUDGET_S = {"gauge": 600, "P0": 1200, "P1": 14400, "P2": 1800, "glue": 1800, "R3": 3600, "R4": 3600,
            "R5": 3600, "R6": 3600, "P6": 900}
EXPECT_H1 = {"ioi": "H1 > 0", "induction": "H1 == 0", "greater_than": "no prediction pre-registered"}


# ============================================================================================== cell plumbing
def cell(status, value=None, criterion="", provenance="", **evidence):
    return dict(status=status, value=value, criterion=criterion, provenance=provenance, evidence=evidence)


def guarded(name):
    def deco(fn):
        def run(*a, **k):
            t = time.time()
            try: c = fn(*a, **k)
            except torch.OutOfMemoryError as e:
                torch.cuda.empty_cache(); c = cell("not_reached", provenance=f"CUDA OOM: {e}")
            except Exception as e:
                c = cell("not_reached", provenance=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])
            w = time.time() - t; c["wall"] = round(w, 1)
            if w > BUDGET_S.get(name, 1e9) and c["status"] != "not_reached":
                c["evidence"]["over_budget"] = dict(wall=round(w, 1), budget_s=BUDGET_S[name])
                c["provenance"] += f" [OVER BUDGET: {w:.0f}s > {BUDGET_S[name]}s pre-registered — kept, flagged]"
            return c
        return run
    return deco


def cv(values):
    v = np.asarray(values, dtype=float)
    return float(v.std(0).max() / (np.abs(v).mean() + 1e-12)) if v.ndim > 1 else float(v.std() / (abs(v.mean()) + 1e-12))


# ============================================================================================== the behaviours
NAMES = ["John", "Mary", "Tom", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
         "Charles", "Thomas", "Daniel", "Paul", "Mark", "Donald", "George", "Kenneth", "Steven", "Edward",
         "Brian", "Ronald", "Anthony", "Kevin", "Jason", "Jeff", "Ryan", "Jacob", "Gary", "Nicholas",
         "Eric", "Stephen", "Jonathan", "Larry", "Justin", "Scott", "Frank", "Brandon", "Ray", "Greg"]
PLACES = ["store", "school", "office", "garden", "hospital", "station", "museum", "restaurant", "park", "house"]
OBJECTS = ["drink", "book", "ring", "kiss", "snack", "bone", "necklace", "basketball", "computer", "letter"]
NOUNS = ["war", "famine", "siege", "plague", "drought", "revolt", "feud", "riot", "strike", "protest"]


def _single(tok, words, prefix=" "):
    """The members of ``words`` that tokenise to exactly one token with ``prefix`` — recorded, never assumed."""
    out = []
    for w in words:
        ids = tok(prefix + w)["input_ids"]
        if len(ids) == 1: out.append((w, ids[0]))
    return out


def build_ioi(tok, n, seed):
    """SPEC_O2 §1.1. Half ABBA, half BABA; fixed token length; metric logit(IO) - logit(S)."""
    rng = np.random.default_rng(seed)
    names = _single(tok, NAMES); places = _single(tok, PLACES); objs = _single(tok, OBJECTS)
    assert len(names) >= 3 and places and objs, "not enough single-token names/places/objects"
    rows, crows, idx, w, meta = [], [], [], [], []
    for i in range(n):
        (A, ai), (B, bi), (C, ci) = [names[j] for j in rng.choice(len(names), 3, replace=False)]
        pl, _ = places[rng.integers(len(places))]; ob, _ = objs[rng.integers(len(objs))]
        abba = i % 2 == 0
        first, second = (A, B) if abba else (B, A)
        clean = f"Then, {first} and {second} went to the {pl}. {B} gave a {ob} to"
        corrupt = f"Then, {first} and {second} went to the {pl}. {C} gave a {ob} to"
        rows.append(tok(clean)["input_ids"]); crows.append(tok(corrupt)["input_ids"])
        idx.append([ai, bi]); w.append([1.0, -1.0]); meta.append(dict(template="ABBA" if abba else "BABA", IO=A, S=B, C=C))
    L = min(len(r) for r in rows)
    T = np.array([r[:L] for r in rows], dtype=np.int64); Tc = np.array([r[:L] for r in crows], dtype=np.int64)
    return dict(name="ioi", tokens=T, corrupt=Tc, idx=np.array(idx, np.int64), w=np.array(w, np.float32),
                meta=meta, seq_len=int(L), n_ragged=int(sum(len(r) != L for r in rows)))


def build_induction(tok, n, seed, L=32, lo=1000, hi=20000):
    """SPEC_O2 §1.2. [BOS] + R + R[:-1]; metric logit(R[-1]) - logit(distractor)."""
    rng = np.random.default_rng(seed); bos = tok.bos_token_id if tok.bos_token_id is not None else 50256
    rows, crows, idx, w = [], [], [], []
    for _ in range(n):
        R = rng.choice(np.arange(lo, hi), size=L, replace=False)
        d = int(rng.integers(lo, hi))
        while d in set(R.tolist()): d = int(rng.integers(lo, hi))
        rows.append([bos] + R.tolist() + R[:-1].tolist())
        R2 = rng.choice(np.arange(lo, hi), size=L - 1, replace=False)       # corrupted: no repeat anywhere
        crows.append([bos] + R.tolist() + R2.tolist())
        idx.append([int(R[-1]), d]); w.append([1.0, -1.0])
    return dict(name="induction", tokens=np.array(rows, np.int64), corrupt=np.array(crows, np.int64),
                idx=np.array(idx, np.int64), w=np.array(w, np.float32), meta=[dict(L=L)] * n, seq_len=2 * L, n_ragged=0)


def build_greater_than(tok, n, seed):
    """SPEC_O2 §1.3. 'The {NOUN} lasted from the year 17{XX} to the year 17'; metric mean logit over YY>XX minus mean over YY<=XX."""
    rng = np.random.default_rng(seed)
    years = []
    for yy in range(100):
        ids = tok(f"{yy:02d}")["input_ids"]
        years.append(ids[0] if len(ids) == 1 else None)
    have = [yy for yy in range(100) if years[yy] is not None]
    nouns = [nnn for nnn, _ in _single(tok, NOUNS, prefix=" ")] or NOUNS
    rows, crows, idx, w, meta = [], [], [], [], []
    K = len(have)
    for _ in range(n):
        xx = int(rng.integers(2, 99)); nn = nouns[rng.integers(len(nouns))]
        rows.append(tok(f"The {nn} lasted from the year 17{xx:02d} to the year 17")["input_ids"])
        crows.append(tok(f"The {nn} lasted from the year 1701 to the year 17")["input_ids"])
        gt = [yy for yy in have if yy > xx]; le = [yy for yy in have if yy <= xx]
        ww = np.zeros(K, np.float32)
        for j, yy in enumerate(have):
            ww[j] = (1.0 / len(gt)) if yy > xx else (-1.0 / max(1, len(le)))
        idx.append([years[yy] for yy in have]); w.append(ww); meta.append(dict(xx=xx, noun=nn, n_gt=len(gt), n_le=len(le)))
    L = min(len(r) for r in rows)
    T = np.array([r[:L] for r in rows], np.int64); Tc = np.array([r[:L] for r in crows], np.int64)
    return dict(name="greater_than", tokens=T, corrupt=Tc, idx=np.array(idx, np.int64), w=np.array(w, np.float32),
                meta=meta, seq_len=int(L), n_ragged=int(sum(len(r) != L for r in rows)),
                n_year_tokens=K, dropped_year_tokens=[yy for yy in range(100) if years[yy] is None])


BUILDERS = {"ioi": build_ioi, "induction": build_induction, "greater_than": build_greater_than}


def make_targets(d, device, sl=None):
    i = torch.as_tensor(d["idx"] if sl is None else d["idx"][sl], device=device)
    w = torch.as_tensor(d["w"] if sl is None else d["w"][sl], device=device)
    return (i, w)


def logitdiff_per_example(logits, targets):
    i, w = targets
    return (logits.gather(1, i) * w).sum(1)


def logitdiff_metric(logits, targets):
    """The one metric of SPEC_O2 §1: m = mean_b sum_k w[b,k] logits[b, idx[b,k]] — a logit difference at one position."""
    return logitdiff_per_example(logits, targets).mean()


# ============================================================================================== the node view
class GPT2NodeView(torch.nn.Module):
    """GPT-2 small as 156 ablatable nodes: 12 layers x (12 attention heads + 1 whole-MLP block), in causal order
    ``node(l, h) = 13 l + h``, ``node(l, mlp) = 13 l + 12`` (SPEC_O2 §0).

    ``forward(tokens, keep=None, means=None, record=False)`` — the protocol ``whiteboxtopo.legibility.discovery``
    expects: ``keep`` a bool mask over nodes (False = replaced by ``means[j]``), logits at the LAST position.

    The recorded head node object is ``z_h`` (B, T, d_head) — the head's output *before* W_O — not the (B, T, d_model)
    projection. The grad x (act - mean) score is identical (head_out = z W_O is linear, so
    (z - mu_z) . dm/dz = (z W_O - mu_z W_O) . dm/dhead_out) and it costs 12x less memory, which is what makes a
    B = 128 recorded forward fit on an 8 GB card. Mean-ablating z is likewise identical to mean-ablating head_out.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model; cfg = model.config
        self.L, self.H, self.d = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.dh = self.d // self.H; self.per = self.H + 1
        self.n_nodes = self.L * self.per
        self.eps = cfg.layer_norm_epsilon
        self.node_names, self.node_layer, self.node_kind = [], [], []
        for l in range(self.L):
            self.node_names += [f"L{l}.h{h}" for h in range(self.H)] + [f"L{l}.mlp"]
            self.node_layer += [l] * self.per; self.node_kind += ["head"] * self.H + ["mlp"]
        self._node_out = []; self.capture_resid = False; self._resid = None
        for p in self.m.parameters(): p.requires_grad_(False)

    # ---- pieces
    def _embed(self, tokens):
        t = self.m.transformer
        return t.wte(tokens) + t.wpe(torch.arange(tokens.shape[1], device=tokens.device))[None]

    def _ln(self, x, ln):
        return Fn.layer_norm(x, (self.d,), ln.weight, ln.bias, self.eps)

    def _block(self, l, x, keep, means, record, outs, freeze_pattern=False):
        blk = self.m.transformer.h[l]; B, T, d = x.shape; H, dh = self.H, self.dh
        base = l * self.per
        h = self._ln(x, blk.ln_1)
        qkv = h @ blk.attn.c_attn.weight + blk.attn.c_attn.bias
        q, k, v = qkv.split(d, dim=2)
        q = q.view(B, T, H, dh).transpose(1, 2); k = k.view(B, T, H, dh).transpose(1, 2); v = v.view(B, T, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        if freeze_pattern: att = att.detach()
        z = att @ v                                                     # (B, H, T, dh)
        Wo = blk.attn.c_proj.weight                                     # (d, d), rows = head*dh
        zs = []
        for hh in range(H):
            zh = z[:, hh]                                               # (B, T, dh)
            if keep is not None and not bool(keep[base + hh]): zh = means[base + hh].to(zh.dtype).expand_as(zh)
            if record:
                zh = zh.clone()
                # an ablated node is a constant: it must still be a graph leaf that requires grad, or
                # ``discovery._scores_under``'s autograd.grad over ALL node outputs raises. Its grad x (act - mean)
                # score is then exactly 0, which is the right score for a node that is not there.
                if not zh.requires_grad: zh.requires_grad_(True)
            zs.append(zh)
        attn_out = sum(zs[hh] @ Wo[hh * dh:(hh + 1) * dh, :] for hh in range(H)) + blk.attn.c_proj.bias
        x = x + attn_out
        h2 = self._ln(x, blk.ln_2)
        mlp_out = blk.mlp.act(h2 @ blk.mlp.c_fc.weight + blk.mlp.c_fc.bias) @ blk.mlp.c_proj.weight + blk.mlp.c_proj.bias
        if keep is not None and not bool(keep[base + H]): mlp_out = means[base + H].to(mlp_out.dtype).expand_as(mlp_out)
        if record:
            mlp_out = mlp_out.clone()
            if not mlp_out.requires_grad: mlp_out.requires_grad_(True)
            outs.extend(zs); outs.append(mlp_out)
        return x + mlp_out

    def forward(self, tokens, keep=None, means=None, record=False, x_embed=None, freeze_pattern=False):
        tokens = torch.as_tensor(tokens, device=next(self.m.parameters()).device)
        if x_embed is None:
            x = self._embed(tokens)
            # every parameter has requires_grad False (the node scores never need weight gradients), so the embedding
            # is made the graph's leaf: without this nothing downstream requires grad and autograd.grad has no graph.
            if torch.is_grad_enabled(): x = x.detach().requires_grad_(True)
        else:
            x = x_embed
        outs = []
        for l in range(self.L):
            x = self._block(l, x, keep, means, record, outs, freeze_pattern=freeze_pattern)
            if self.capture_resid and l == LAYER: self._resid = x
        if record: self._node_out = outs
        x = self._ln(x, self.m.transformer.ln_f)
        return (x[:, -1] @ self.m.transformer.wte.weight.T)

    # ---- the discovery protocol
    @torch.no_grad()
    def node_means(self, tokens, batch=64):
        """Per-node mean output over ``tokens``, per position: heads (T, dh), MLP blocks (T, d). SPEC_O2 §1.4."""
        acc = None; n = 0
        for i in range(0, len(tokens), batch):
            self.forward(tokens[i:i + batch], record=True)
            s = [o.detach().sum(0) for o in self._node_out]
            acc = s if acc is None else [a + b for a, b in zip(acc, s)]
            n += len(tokens[i:i + batch]); self._node_out = []
        return [a / n for a in acc]

    def node_scores(self, tokens, targets, means, metric=logitdiff_metric, batch=None):
        """(n_nodes,) attribution-patching scores sum_batch (node - mean) . dm/dnode. Chunked: the metric is a batch
        mean, so each chunk's contribution is weighted by its share (the convention of ``e4models.edge_scores``)."""
        B = len(tokens); batch = batch or B; out = torch.zeros(self.n_nodes)
        for i in range(0, B, batch):
            sl = slice(i, min(i + batch, B)); wgt = (sl.stop - sl.start) / B
            self.m.zero_grad(set_to_none=True)
            logits = self.forward(tokens[sl], record=True)
            mval = metric(logits, (targets[0][sl], targets[1][sl]))
            g = torch.autograd.grad(mval, self._node_out, allow_unused=True)
            for j, o in enumerate(self._node_out):
                if g[j] is not None: out[j] += wgt * ((o - means[j]) * g[j]).sum().item()
            self._node_out = []
        return out

    @torch.no_grad()
    def resid(self, tokens, keep=None, means=None, batch=32, last_only=False):
        """Mid-layer (block LAYER) residual stream; (N, T, d) or (N, d) at the last position."""
        self.capture_resid = True; outs = []
        try:
            for i in range(0, len(tokens), batch):
                self.forward(tokens[i:i + batch], keep=keep, means=means)
                r = self._resid.detach()
                outs.append((r[:, -1] if last_only else r).float().cpu())
        finally: self.capture_resid = False; self._resid = None
        return torch.cat(outs).numpy()

    @torch.no_grad()
    def metric_per_example(self, tokens, targets, keep=None, means=None, batch=32):
        out = []
        for i in range(0, len(tokens), batch):
            sl = slice(i, min(i + batch, len(tokens)))
            lg = self.forward(tokens[sl], keep=keep, means=means)
            out.append(logitdiff_per_example(lg, (targets[0][sl], targets[1][sl])).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def gathered_logits(self, tokens, targets, keep=None, means=None, batch=32):
        out = []
        for i in range(0, len(tokens), batch):
            sl = slice(i, min(i + batch, len(tokens)))
            lg = self.forward(tokens[sl], keep=keep, means=means)
            out.append(lg.gather(1, targets[0][sl]).float().cpu())
        return torch.cat(out).numpy()


def faith(view, tokens, targets, keep, means, m_clean, m_corr, batch=32):
    """Gap-normalised faithfulness, chunked (``discovery.faithfulness`` in one shot would not fit at T = 64)."""
    v = float(view.metric_per_example(tokens, targets, keep=keep, means=means, batch=batch).mean())
    return (v - m_corr) / (m_clean - m_corr) if abs(m_clean - m_corr) > 1e-8 else float("nan")


def head_align_map(view, g):
    """Base node index of each node of the transformed model: transformed node j == base node p[j] for a head
    permutation p (``symmetry.gpt2.apply``); MLP-block nodes are fixed by every element of G (SPEC_O2 §0)."""
    a = np.arange(view.n_nodes)
    for l in range(view.L):
        p = g.perms.get(("head", l))
        if p is None: continue
        p = np.asarray(p); a[l * view.per:l * view.per + view.H] = l * view.per + p
    return a


# ============================================================================================== step 1: gauge
@guarded("gauge")
def cell_gauge(model, view, d, dev, n_gauge):
    from whiteboxtopo.symmetry import gpt2 as g2
    T = torch.as_tensor(d["tokens"][:8], device=dev)
    with torch.no_grad(): ref = model(T).logits.float()
    scale = float(ref.abs().max()); rng = np.random.default_rng(11); devs = {}
    for kind in ("perm", "scale", "rot", "combined"):
        worst = 0.0
        for r in range(min(5, max(2, n_gauge))):
            g = g2.sample(model, rng, kind=kind, orthogonal=(r % 2 == 0))
            mm = g2.apply(model, g)
            with torch.no_grad(): worst = max(worst, float((mm(T).logits.float() - ref).abs().max() / scale))
            del mm
        devs[kind] = worst
    # approximate symmetries, measured not sampled (FINAL_SPEC 2.3): MLP scaling through GELU, and ln_f rescaling
    approx = {}
    mm = copy.deepcopy(model); a = 1.5
    with torch.no_grad():
        for blk in mm.transformer.h:
            blk.mlp.c_fc.weight.mul_(a); blk.mlp.c_fc.bias.mul_(a); blk.mlp.c_proj.weight.div_(a)
        approx["mlp_scaling_through_gelu"] = float((mm(T).logits.float() - ref).abs().max() / scale)
    del mm
    mm = copy.deepcopy(model)
    with torch.no_grad():
        mm.transformer.ln_f.weight.mul_(a); mm.transformer.ln_f.bias.mul_(a)
        approx["ln_f_rescale_uncompensated"] = float((mm(T).logits.float() - ref).abs().max() / scale)
    del mm; torch.cuda.empty_cache()
    # the node view must reproduce HF exactly (it is a hand-written forward)
    with torch.no_grad(): vdev = float((view.forward(T) - ref[:, -1]).abs().max() / scale)
    ok = max(devs.values()) < ATTR_CV and vdev < 1e-5
    return cell("certified" if ok else "failed", value=dict(max_rel_dev=devs, node_view_rel_dev=vdev),
                criterion=f"every exact kind (perm/scale/rot/combined) has max relative |dlogit| < {ATTR_CV} (fp32 realisation of 'exact'); the hand-written node-view forward reproduces GPT2LMHeadModel to < 1e-5 relative",
                provenance="whiteboxtopo.symmetry.gpt2: head permutations, within-head GL(d_h) rotations (Q R, K R^-T, V R, R^-1 O), MLP hidden-unit permutations, ln_1/ln_2 gain rescalings; 5 samples per kind on 8 task prompts",
                group=g2.describe(model), approximate_symmetry_residual=approx,
                hidden_symmetries="not_reached: dim ker dR (Jacobian corank, FINAL_SPEC 2.2) is out of reach for 124 M parameters on this hardware; every certification below is against the KNOWN group G")


# ============================================================================================== step 2: P0
def attribution_graph(view, tokens, targets, means, metric=logitdiff_metric):
    """E[i, j] = |sum_batch (node_i - mean_i) . d s_j / d node_i| with s_j node j's own grad x (act - mean) score,
    and E[j, out] = |s_j|; edges only from strictly upstream nodes. The GPT-2 lift of ``olib.attribution_graph``."""
    n = view.n_nodes; E = np.zeros((n + 1, n + 1))
    view.m.zero_grad(set_to_none=True)
    logits = view.forward(tokens, record=True); outs = view._node_out
    g = torch.autograd.grad(metric(logits, targets), outs, retain_graph=True, allow_unused=True)
    s = [None if g[j] is None else ((outs[j] - means[j]) * g[j]).sum() for j in range(n)]
    for j in range(n):
        if s[j] is None: continue
        E[j, n] = abs(s[j].item())
        if j == 0: continue
        gh = torch.autograd.grad(s[j], outs[:j], retain_graph=True, allow_unused=True)
        for i in range(j):
            if gh[i] is not None: E[i, j] = abs(((outs[i] - means[i]) * gh[i]).sum().item())
    view._node_out = []
    return E, view.node_names + ["logits"]


@guarded("P0")
def cell_P0(view, tokens, targets, means, seed=0, batch=64):
    from whiteboxtopo.legibility import graph as G
    n = min(batch, len(tokens))
    E, names = attribution_graph(view, tokens[:n], (targets[0][:n], targets[1][:n]), means)
    A = G.threshold_density(G.symmetrise(E), 0.1)
    Q, _ = G.modularity_q(A, seed=seed); tw = int(G.treewidth_bound(A)); sp = float(G.sparsity(np.abs(E).sum(1)))
    null = G.shuffled_q_null(E, density=0.1, n=20, seed=seed)
    qn = np.asarray(null["Q"] if isinstance(null, dict) else null, float)
    verdict = "tractable" if tw <= 8 else ("marginal" if tw <= 32 else "intractable")
    return cell("certified" if verdict == "tractable" else "failed", value=verdict,
                criterion="tractable iff treewidth_ub <= 8 at density 0.1 (the FPT parameter, FINAL_SPEC 2.14; AMENDMENTS #5d), marginal <= 32, else intractable; Q reported against the rewiring null but does not decide the verdict",
                provenance=f"attribution graph over the {view.n_nodes} head+MLP-block nodes on {n} prompts (grad x (act - mean), causal edges only) + legibility.graph; null = degree-preserving rewiring (20 draws)",
                Q=float(Q), treewidth=tw, sparsity=sp, null_Q_mean=float(qn.mean()), null_Q_max=float(qn.max()),
                Q_minus_null=float(Q) - float(qn.mean()), n_nodes=view.n_nodes, n_prompts=n)


# ============================================================================================== steps 3-4: P1
def _random_set_faith(view, tokens, targets, means, size, m_clean, m_corr, n=20, seed=0, batch=32):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        k = torch.zeros(view.n_nodes, dtype=torch.bool)
        k[torch.as_tensor(rng.choice(view.n_nodes, size=size, replace=False))] = True
        vals.append(faith(view, tokens, targets, k, means, m_clean, m_corr, batch=batch))
    v = np.asarray(vals, float)
    return dict(mean=float(np.nanmean(v)), q99=float(np.nanquantile(v, 0.99)), max=float(np.nanmax(v)), n=n)


@guarded("P1")
def cell_P1(model, view, d, dev, tokens, targets, means, n_gauge, batch=32, seed=0):
    from whiteboxtopo.legibility.discovery import discover_circuit, jaccard
    from whiteboxtopo.symmetry import gpt2 as g2
    m_clean = float(view.metric_per_example(tokens, targets, batch=batch).mean())
    none = torch.zeros(view.n_nodes, dtype=torch.bool)
    m_corr = float(view.metric_per_example(tokens, targets, keep=none, means=means, batch=batch).mean())
    r = discover_circuit(view, tokens, targets, means, seed=seed, F_target=F_TARGET, metric=logitdiff_metric)
    S = set(int(i) for i in np.flatnonzero(r["keep"].numpy()))
    if not S: return cell("failed", value=dict(size=0), criterion="discover_circuit must reach F >= 0.9", provenance="empty circuit")
    nodes = sorted(S); names = [view.node_names[i] for i in nodes]
    null = _random_set_faith(view, tokens, targets, means, len(S), m_clean, m_corr, n=20, seed=seed, batch=batch)
    # K-O-2 at T2: the circuit re-derived on the gauge orbit
    rng = np.random.default_rng(23); Fg, same, jac, inter = [], [], [], set(S)
    for gi in range(n_gauge):
        g = g2.sample(model, rng, kind="combined")
        mm = g2.apply(model, g); vv = GPT2NodeView(mm).to(dev).eval()
        try:
            a = head_align_map(vv, g)                                   # transformed node j == base node a[j]
            mg = vv.node_means(tokens, batch=batch)
            keep_g = torch.zeros(vv.n_nodes, dtype=torch.bool)
            for j in range(vv.n_nodes):
                if int(a[j]) in S: keep_g[j] = True
            mc = float(vv.metric_per_example(tokens, targets, batch=batch).mean())
            mz = float(vv.metric_per_example(tokens, targets, keep=torch.zeros(vv.n_nodes, dtype=torch.bool), means=mg, batch=batch).mean())
            Fg.append(faith(vv, tokens, targets, keep_g, mg, mc, mz, batch=batch))
            rg = discover_circuit(vv, tokens, targets, mg, seed=seed, F_target=F_TARGET, metric=logitdiff_metric)
            Sg = set(int(a[j]) for j in np.flatnonzero(rg["keep"].numpy()))
            same.append(Sg == S); inter &= Sg
            jac.append(len(Sg & S) / max(1, len(Sg | S)))
        finally:
            del vv, mm; torch.cuda.empty_cache()
    cvF = cv(Fg) if Fg else float("nan")
    frac = len(inter) / len(S)
    beats = r["F"] > null["q99"]
    ok = bool(Fg) and all(same) and cvF < ART_CV and frac == 1.0 and beats
    status = "certified" if ok else ("partial" if (frac > 0 and beats) else "failed")
    return cell(status, value=dict(size=len(S), frac_of_model=len(S) / view.n_nodes, F=r["F"], certified_fraction=frac,
                                   node_sets_identical=bool(Fg) and all(same), cv_faithfulness=cvF),
                criterion=f"circuit = discover_circuit at faithfulness {F_TARGET} with mean ablation (frozen pipeline); certified iff the gauge-aligned node set is identical on all {n_gauge} gauge samples AND CV(faithfulness) < {ART_CV} AND the certified fraction is 1.0 AND F exceeds the 99th percentile of {null['n']} random node sets of the same size; partial if the fraction is < 1",
                provenance=f"legibility.discovery.discover_circuit (2 passes, rescore_every 16, seed {seed}) on {len(tokens)} prompts; gauge orbit = symmetry.gpt2.sample(kind='combined'), means recomputed on every transformed model, node indices aligned through the head permutation",
                nodes=nodes, node_names=names, cost_forward_passes=int(r["cost"]), gap=float(r["gap"]),
                m_clean=m_clean, m_corr=m_corr, F_gauge=[float(x) for x in Fg], jaccard_gauge=jac,
                certified_nodes=sorted(inter), certified_node_names=[view.node_names[i] for i in sorted(inter)],
                null_random_node_set=null, beats_random_null=bool(beats),
                ko1_ko3="not applicable at T2: no irrep inventory to randomise (K-O-1) and no memorising control specimen at GPT-2 scale (K-O-3); K-O-2 is the gauge check above")


# ============================================================================================== P2
def token_contributions(view, tokens, targets, freeze_pattern=True):
    """Frozen-pattern contribution of each input token embedding to the metric: c[b, t] = (dm/dx[b,t]) . e[b,t].
    QK caveat: with the pattern detached this is exact for the OV circuit and ignores the query-side dependence."""
    t = view.m.transformer
    e = t.wte(tokens).detach()                                  # the token embedding: the "input" being attributed
    x = (e + t.wpe(torch.arange(tokens.shape[1], device=tokens.device))[None]).detach().requires_grad_(True)
    m = logitdiff_metric(view.forward(tokens, x_embed=x, freeze_pattern=freeze_pattern), targets)
    g, = torch.autograd.grad(m, x)
    return (g * e).sum(-1).detach()


def integrated_gradients(view, tokens, targets, steps=32, delta=0.05, freeze_pattern=False):
    """IG of the metric along the constant-pattern path (1-delta) e -> e (AMENDMENTS #5c), scaled to the full
    contribution. ``freeze_pattern`` selects which model the IG is taken through: the pattern-frozen linearisation
    (the matched anchor for the frozen-pattern path product, AMENDMENTS #O2-1) or the full model."""
    t = view.m.transformer
    pos = t.wpe(torch.arange(tokens.shape[1], device=tokens.device))[None]
    e = t.wte(tokens).detach(); ig = torch.zeros_like(e)
    for s in range(1, steps + 1):
        x = (pos + e * (1 - delta + delta * s / steps)).detach().requires_grad_(True)
        m = logitdiff_metric(view.forward(tokens, x_embed=x, freeze_pattern=freeze_pattern), targets)
        g, = torch.autograd.grad(m, x); ig += g
    return (ig / steps * e).sum(-1).detach()


@guarded("P2")
def cell_P2(model, view, dev, tokens, targets, n_gauge, batch=32):
    from whiteboxtopo.symmetry import gpt2 as g2
    n = min(batch, len(tokens)); tk = torch.as_tensor(tokens[:n], device=dev); tg = (targets[0][:n], targets[1][:n])
    c = token_contributions(view, tk, tg).cpu().numpy()
    rng = np.random.default_rng(31); orbit = []
    for _ in range(min(n_gauge, 20)):
        mm = g2.apply(model, g2.sample(model, rng, kind="combined")); vv = GPT2NodeView(mm).to(dev).eval()
        try: orbit.append(token_contributions(vv, tk, tg).cpu().numpy())
        finally: del vv, mm; torch.cuda.empty_cache()
    cv_attr = cv(orbit) if orbit else float("nan")
    cfull = token_contributions(view, tk, tg, freeze_pattern=False).cpu().numpy()
    ig_frozen = integrated_gradients(view, tk, tg, delta=0.05, freeze_pattern=True).cpu().numpy()
    ig_full = integrated_gradients(view, tk, tg, delta=0.05, freeze_pattern=False).cpu().numpy()
    ig_g = integrated_gradients(view, tk, tg, delta=1.0, freeze_pattern=False).cpu().numpy()
    rr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    r_anchor = rr(c, ig_frozen)                       # the pre-registered anchor, on the constant-pattern region
    r_qk = rr(c, ig_full)                             # the QK caveat, quantified
    ok = cv_attr < ATTR_CV and r_anchor > 0.95
    return cell("certified" if ok else ("partial" if cv_attr < ATTR_CV else "failed"),
                value=dict(cv=cv_attr, ig_r_anchor_frozen=r_anchor, ig_r_full_model=r_qk,
                           ig_r_global=rr(c, ig_g), qk_share_r_frozen_vs_full=rr(c, cfull)),
                criterion=f"attention-pattern-conditional path products: CV over the gauge orbit < {ATTR_CV} (the fp32 realisation of FINAL_SPEC 6.3's CV < 1e-6, which fp32 GPT-2 logits cannot reach — recorded as a precision limit, not a model property); IG anchor r > 0.95 on the constant-pattern path (1-0.05)e -> e, with the IG taken through the SAME pattern-frozen linearisation (AMENDMENTS #O2-1). The full-model IG correlation is reported as the measured size of the QK term, not as the anchor",
                provenance=f"frozen-pattern (attention probabilities detached) Jacobian contributions of each input token embedding to the logit difference, {n} prompts; gauge orbit n={len(orbit)} (symmetry.gpt2 combined); IG 32 steps; the full-gradient attribution and the zero-baseline global IG reported as diagnostics",
                qk_caveat="the attribution is exact for the OV circuit at frozen pattern and ignores the query-side dependence of the pattern on the input (FINAL_SPEC 6.3); ig_r_full_model / qk_share_r_frozen_vs_full measure how large that omission is on this behaviour",
                top_token_share=float(np.mean(np.abs(c).max(1) / (np.abs(c).sum(1) + 1e-9))))


# ============================================================================================== step 6: glue
def per_example_scores(view, tokens, targets, means, batch=16):
    """(N, n_nodes) per-example explanation atoms: sum over positions of grad x (act - mean) per node."""
    out = []
    for i in range(0, len(tokens), batch):
        sl = slice(i, min(i + batch, len(tokens)))
        view.m.zero_grad(set_to_none=True)
        logits = view.forward(tokens[sl], record=True); outs = view._node_out
        per = logitdiff_per_example(logits, (targets[0][sl], targets[1][sl]))
        g = torch.autograd.grad(per.sum(), outs, allow_unused=True)
        cols = []
        for j, o in enumerate(outs):
            if g[j] is None: cols.append(torch.zeros(o.shape[0], device=o.device)); continue
            cols.append(((o - means[j]) * g[j]).reshape(o.shape[0], -1).sum(1))
        out.append(torch.stack(cols, 1).detach().float().cpu()); view._node_out = []
    return torch.cat(out).numpy()


def standardise(X):
    X = np.asarray(X, dtype=np.float64); sd = X.std(0); keep = sd > 1e-8
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep], keep


def _nerve_components(cov):
    """Connected components of the nerve graph (regions as vertices, overlaps >= n_min as edges)."""
    k = cov.n_regions; parent = list(range(k))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in cov.edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj: parent[ri] = rj
    return len({find(v) for v in range(k)})


@guarded("glue")
def cell_glue(view, task, tokens, targets, means, k=COVER_K, seed=0):
    from whiteboxtopo.sheaf import cover, cech_H0_H1
    Z, keep = standardise(view.resid(tokens, last_only=True))
    n = len(Z)
    # k = 10 and n_min = 10 are the pre-registered values (SPEC_O2 §6) and are used at every full-run size;
    # a --smoke split is too small for them, so both are reduced and the reduction is recorded in the cell.
    k_eff = int(min(k, max(3, n // 12))); n_min = int(min(10, max(3, n // 8)))
    cov = cover(Z, k=k_eff, overlap=0.5, seed=seed, n_min=n_min)
    E = per_example_scores(view, tokens, targets, means)
    res = cech_H0_H1(cov, E, tau=GLUE_TAU, k_e=GLUE_KE)
    rng = np.random.default_rng(seed)
    E0 = np.tile(E.mean(0, keepdims=True), (len(E), 1)) + 1e-9 * rng.standard_normal(E.shape)
    res0 = cech_H0_H1(cov, E0, tau=GLUE_TAU, k_e=GLUE_KE)
    # the trivial explanation makes every pair of regions agree, so its H0 is the number of CONNECTED COMPONENTS of
    # the nerve, which is 1 only when the nerve is connected. A disconnected nerve (few prompts, k regions) is a
    # property of the cover, not a failure of the instrument, so the null is stated against the component count.
    comp = _nerve_components(cov)
    null_ok = res0.H1 == 0 and res0.H0 == comp
    informative = res.cyclomatic > 0
    exp = EXPECT_H1[task]
    met = (res.H1 > 0) if task == "ioi" else ((res.H1 == 0) if task == "induction" else None)
    # boundary localisation: which nodes the worst disagreement is about
    bd = sorted(res.boundaries, key=lambda b: b["J"])[:5]
    loc = []
    for b in bd:
        mi, mj = cov.members[b["i"]], cov.members[b["j"]]
        ti = set(np.argsort(-np.abs(E[mi]).mean(0))[:GLUE_KE].tolist()); tj = set(np.argsort(-np.abs(E[mj]).mean(0))[:GLUE_KE].tolist())
        loc.append(dict(i=b["i"], j=b["j"], J=b["J"], J_bott=b["J_bott"], frustrated=b["frustrated"],
                        only_i=[view.node_names[x] for x in sorted(ti - tj)], only_j=[view.node_names[x] for x in sorted(tj - ti)]))
    if not null_ok: status = "failed"
    elif not informative: status = "partial"
    elif met is None: status = "partial"
    else: status = "certified" if met else "failed"
    return cell(status, value=dict(H0=int(res.H0), H1=int(res.H1), expectation=exp,
                                   expectation_met=met, cyclomatic=int(res.cyclomatic)),
                criterion=f"cover k={k_eff} (pre-registered {k}) overlap 0.5 of the task inputs by the per-unit standardised mid-layer residual cloud; per-example atoms = per-node grad x (act - mean); cech_H0_H1 tau {GLUE_TAU} k_e {GLUE_KE}. Pre-stated expectation: IOI H1 > 0, induction H1 == 0 (greater-than: none). Trivial-explanation null must give H0 = 1, H1 = 0 or the instrument is reported as failing its own null. cyclomatic == 0 -> H1 = 0 by construction, not evidence (partial).",
                provenance=f"sheaf.cover(knn2, k={k_eff}, n_min={n_min}; pre-registered k={k}, n_min=10 — reduced only when the split is too small, recorded here) on the standardised layer-{LAYER} residual stream at the last position of {len(tokens)} prompts ({int(keep.sum())} live units; the standardisation is the recorded gauge fixing); atoms over the {view.n_nodes} head+MLP-block nodes; induction carries the sanity gate at T2 (AMENDMENTS #5f: H1 > 0 on the single-mechanism task indicts the instrument, not the model)",
                cover=cov.to_dict(), cover_k=k_eff, cover_k_prereg=k, cover_n_min=n_min, cech={kk: v for kk, v in res.to_dict().items() if kk not in ("J", "J_bott")},
                trivial_null=dict(H0=int(res0.H0), H1=int(res0.H1), nerve_components=int(comp), passed=bool(null_ok)),
                boundaries_n=len(res.boundaries), boundary_localisation=loc)


# ============================================================================================== steps 7-8: R cells
@guarded("R3")
def cell_R3(view, p1, ho_tokens, ho_targets, means, batch=32, seed=0):
    if p1["status"] in ("not_reached", "failed"): return cell("not_reached", provenance=f"no certified circuit (P1 {p1['status']})")
    S = torch.zeros(view.n_nodes, dtype=torch.bool); S[p1["evidence"]["nodes"]] = True
    m_clean = float(view.metric_per_example(ho_tokens, ho_targets, batch=batch).mean())
    none = torch.zeros(view.n_nodes, dtype=torch.bool)
    m_corr = float(view.metric_per_example(ho_tokens, ho_targets, keep=none, means=means, batch=batch).mean())
    F = faith(view, ho_tokens, ho_targets, S, means, m_clean, m_corr, batch=batch)
    null = _random_set_faith(view, ho_tokens, ho_targets, means, int(S.sum()), m_clean, m_corr, n=20, seed=seed, batch=batch)
    ok = F >= F_TARGET and F > null["q99"]
    return cell("certified" if ok else "failed", value=dict(faithfulness_heldout=F, size=int(S.sum())),
                criterion=f"R3: f_hat = the certified circuit with the complement mean-ablated; faithfulness >= {F_TARGET} on held-out prompts, and above the 99th percentile of {null['n']} random node sets of the same size",
                provenance=f"the node view run with keep = the certified node set and the discovery-split means, on {len(ho_tokens)} held-out prompts. reconstruct.from_circuits is NOT used: it calls invariants.model.as_relu_mlp and is a ReLU-MLP instrument (its own docstring states this failure boundary); R3 here is the same DEFINITION on the GPT-2 node view",
                m_clean=m_clean, m_corr=m_corr, null_random_node_set=null)


@guarded("R4")
def cell_R4(view, p1, ho_tokens, means, batch=16, seed=0):
    from whiteboxtopo.reconstruct import residual, coverage_gap
    if p1["status"] in ("not_reached", "failed"): return cell("not_reached", provenance=f"no certified circuit (P1 {p1['status']})")
    S = torch.zeros(view.n_nodes, dtype=torch.bool); S[p1["evidence"]["nodes"]] = True
    H = view.resid(ho_tokens, batch=batch).reshape(-1, view.d)
    Hh = view.resid(ho_tokens, keep=S, means=means, batch=batch).reshape(-1, view.d)
    sd = H.std(0); live = sd > 1e-8
    Hs = (H[:, live] - H[:, live].mean(0)) / sd[live]; Hhs = (Hh[:, live] - H[:, live].mean(0)) / sd[live]
    R = residual(Hs, Hhs); cg = coverage_gap(R, q=0.99, seed=seed); dd = cg.to_dict()
    r2 = 1 - float((R ** 2).sum() / ((Hs - Hs.mean(0)) ** 2).sum())
    clear = dd["gamma_cov"] > max(dd.get("gamma_null_gaussian", 0.0), dd.get("gamma_null_shuffled", 0.0))
    return cell("partial" if clear else "failed", value=dict(gamma_cov=dd["gamma_cov"], gamma_excess=dd.get("gamma_excess"),
                                                             degree_of_opening=1 - dd["gamma_cov"], r2_circuit=r2),
                criterion="R4: gamma_cov of the residual between the full model's mid-layer activations and the circuit-only run's, via the K-block MP null (reconstruct.coverage_gap, Tracy-Widom-corrected edge q 0.99). FINAL_SPEC 7 R4 sets NO pass threshold at T2 ('T2-T4: report'), so the status is 'partial' whenever a number is produced and 'failed' only if the gap is not clear of its own nulls (Gaussian matched-marginal and coordinate-shuffled).",
                provenance=f"h = per-unit standardised (recorded gauge fixing) layer-{LAYER} residual stream, all positions of {len(ho_tokens)} held-out prompts ({R.shape[0]} x {R.shape[1]}); h_hat = the same activations with the circuit's complement mean-ablated",
                coverage=dd, n_samples=int(R.shape[0]), n_units=int(live.sum()))


@guarded("R5")
def cell_R5(view, p1, ho_tokens, ho_targets, means, batch=32):
    from whiteboxtopo.verify import equivalence
    if p1["status"] in ("not_reached", "failed"): return cell("not_reached", provenance=f"no certified circuit (P1 {p1['status']})")
    S = torch.zeros(view.n_nodes, dtype=torch.bool); S[p1["evidence"]["nodes"]] = True
    m = view.metric_per_example(ho_tokens, ho_targets, batch=batch)
    mh = view.metric_per_example(ho_tokens, ho_targets, keep=S, means=means, batch=batch)
    r = float(np.corrcoef(m, mh)[0, 1]); band = abs(float(mh.mean()) - float(m.mean())) <= 0.1 * abs(float(m.mean()))
    L = view.gathered_logits(ho_tokens, ho_targets, batch=batch); Lh = view.gathered_logits(ho_tokens, ho_targets, keep=S, means=means, batch=batch)
    X = ho_tokens.detach().cpu().numpy() if torch.is_tensor(ho_tokens) else np.asarray(ho_tokens)
    v = equivalence(lambda _: L, lambda _: Lh, X, radii=()).to_dict()
    ok = r >= 0.9 and band
    return cell("certified" if ok else "failed", value=dict(logitdiff_r=r, mean_m=float(m.mean()), mean_m_hat=float(mh.mean()),
                                                            rel_gap=abs(float(mh.mean()) - float(m.mean())) / (abs(float(m.mean())) + 1e-12)),
                criterion="R5: Pearson r of the per-example logit difference f_hat vs f on held-out prompts >= 0.9 AND |mean m(f_hat) - mean m(f)| <= 0.1 |mean m(f)|. The Gaussian/PGD perturbation curves and the OOD battery of FINAL_SPEC 7 R5 are NOT APPLICABLE: f_hat is the same network with nodes mean-ablated, so it shares f's input space and a perturbation curve would measure the model, not the rebuild — recorded, not counted",
                provenance="the node view at keep = certified circuit vs keep = all, on the held-out split; verify.equivalence on the gathered metric logits reported alongside as a diagnostic",
                equivalence=v, not_applicable=["gaussian_perturbation", "pgd", "ood_agreement"])


@guarded("R6")
def cell_R6(view, p1, ho_tokens, ho_targets, means, batch=32):
    from whiteboxtopo.verify import ablation_symmetry
    if p1["status"] in ("not_reached", "failed"): return cell("not_reached", provenance=f"no certified circuit (P1 {p1['status']})")
    nodes = list(p1["evidence"]["nodes"])
    S = torch.zeros(view.n_nodes, dtype=torch.bool); S[nodes] = True
    full = torch.ones(view.n_nodes, dtype=torch.bool)

    def f_ab(k, X):
        kk = full.clone()
        if k is not None: kk[k] = False
        return view.metric_per_example(X, ho_targets, keep=kk, means=means, batch=batch)

    def fh_ab(k, X):
        kk = S.clone()
        if k is not None: kk[k] = False
        return view.metric_per_example(X, ho_targets, keep=kk, means=means, batch=batch)

    tab = ablation_symmetry(f_ab, fh_ab, nodes, ho_tokens, threshold=0.9); dd = tab.to_dict()
    rs = [r["r"] for r in dd["rows"]]
    return cell("certified" if dd["all_passed"] else "failed",
                value=dict(n_objects=len(nodes), n_failed=dd["n_failed"], r_min=float(np.nanmin(rs)) if rs else None),
                criterion="R6: for each certified node, ablate it in f (full model) and in f_hat (circuit); the per-example effect vectors on the held-out prompts must correlate r > 0.9. Any node below is mis-mapped and is named",
                provenance="verify.ablation_symmetry; effect vector = the change in the per-example logit difference",
                table=dd, failed_nodes=[view.node_names[r["object"]] for r in dd["rows"] if not r["passed"]])


# ============================================================================================== P6
@guarded("P6")
def cell_P6(view, tokens, seed=0):
    from whiteboxtopo.signals import memorization_signal
    A = view.resid(tokens, last_only=True).astype(np.float64)
    s = memorization_signal(activations=A, family="lm_transformer", seed=seed)
    d = s.to_dict() if hasattr(s, "to_dict") else dict(score=float(s.score), ci_low=float(s.ci_low), ci_high=float(s.ci_high))
    sc = d.get("score", getattr(s, "score", None))
    sat = sc is not None and (sc > 1 - 1e-6 or sc < 1e-6)
    return cell("partial", value=dict(score=sc, saturated=bool(sat),
                                      ci=[d.get("ci_low", getattr(s, "ci_low", None)), d.get("ci_high", getattr(s, "ci_high", None))],
                                      n_probe=d.get("n_probe", getattr(s, "n_probe", None))),
                criterion="P6 at T2 (FINAL_SPEC 6.7): the audited M-block memorization_signal for the lm_transformer family with its shipped calibration, reported honestly against baselines. It can NEVER be 'certified' here: 6.7's pass criterion is a SEPARATION (AUC) between memorising and generalising models, and T2 has one model and one family — there is no contrast, so there is no AUC. Reporting a single calibrated score as a verdict is what 12.3 forbids",
                provenance=f"signals.memorization_signal on the layer-{LAYER} residual cloud at the last position of {len(tokens)} task prompts; calibration whiteboxtopo/signals/calibration/memorization_lm_transformer.json (M block, audited in M1/M2)",
                signal=d,
                saturation_note=("the calibrated score is at the numerical edge of its range, so it does not resolve this cloud at all: "
                                 "two different behaviours' clouds return the same value to every printed digit. The M-block lm_transformer "
                                 "logistic was calibrated on a battery of memorising vs generalising LMs, and GPT-2's residual cloud sits "
                                 "off that scale. Recorded as the measurement it is — a saturated instrument, not a verdict about GPT-2") if sat else None)


# ============================================================================================== not-reached cells
def not_reached_cells():
    return {
        "P3": cell("not_reached", provenance="FINAL_SPEC 6.4 needs either a planted shortcut with a known onset (T1) or training checkpoints (T3). GPT-2 ships one final checkpoint: there is no trajectory to run zigzag or Laplacian eigenflow over. P3 at a real training run is O3.1 (Pythia, HPCC)",
                   criterion="not run; no checkpoints"),
        "P4": cell("not_reached", provenance="FINAL_SPEC 6.5's diffusion-embedding OOD predictor FAILED its pre-registration at T1 (O1.4 Fine Motion: AUC 0.41-0.51 vs the required > 0.9, beaten by softmax confidence at 0.999). Operating Constitution 12.1 forbids re-running a failed procedure at a new tier without a new pre-registration, and no such pre-registration exists",
                   criterion="not run; failed at T1, not re-registered"),
        "P5": cell("not_reached", provenance="FINAL_SPEC 6.6's curvature x Lipschitz failure predictor FAILED at T1 (O1.4: Spearman 0.10, p 0.09 against the required < -0.5 at p < 0.01). It also needs the exact path product A(x), which at GPT-2 exists only in the frozen-pattern linearisation of P2 — the quantity itself is not the pre-registered one",
                   criterion="not run; failed at T1, not re-registered"),
        "P7": cell("not_reached", provenance="No training checkpoints, as for P3. FINAL_SPEC 8.2 assigns P7 on a real training run to O3.1 (Pythia-410M/1B, 154 public checkpoints, HPCC)",
                   criterion="not run; no checkpoints"),
        "R7": cell("not_reached", provenance="R7's artifact-class rebuild has no meaning for a node circuit: the 'artifact class' is just the complement of the certified node set, whose faithfulness is already the random-node-set null of P1/R3",
                   criterion="not run; pre-registered in SPEC_O2 7"),
        "R8": cell("not_reached", provenance="R8 is gated on R5 failing AND on a second extraction round the T2 schedule does not fund",
                   criterion="not run; pre-registered in SPEC_O2 7"),
    }


# ============================================================================================== drivers
def scorecard(name, cells, behavior):
    from whiteboxtopo.open import Scorecard, CellResult
    card = Scorecard(model=name, behavior=behavior)
    for p in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        c = cells.get(p)
        if c is None: card.cells[p] = CellResult(p, "not_reached", provenance="not run"); continue
        st = c["status"] if c["status"] in ("certified", "partial", "failed", "not_reached") else "failed"
        card.cells[p] = CellResult(p, st, value=c["value"], evidence={"criterion": c["criterion"]}, provenance=c["provenance"])
    r4 = cells.get("R4"); g = None
    if r4 and r4["status"] != "not_reached" and isinstance(r4.get("value"), dict):
        g = r4["value"].get("gamma_cov")
        if isinstance(g, dict): g = None            # the merged card aggregates per-task values; set explicitly by merge()
    card.gamma_cov = g
    card.reconstruction = {k: cells[k]["status"] for k in cells if k.startswith("R")}
    return card


def write(task, out, behavior):
    p = os.path.join(OUT, f"satono_crown_{task}.json")
    card = scorecard(f"gpt2-small [{task}]", out["cells"], behavior)
    out["scorecard"] = card.to_dict()
    json.dump(out, open(p, "w"), indent=1, default=str)
    open(os.path.join(OUT, f"satono_crown_{task}.md"), "w").write(card.to_markdown() + "\n")
    new = not os.path.exists(CELLS_CSV)
    with open(CELLS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS); new and w.writeheader()
        for k, c in out["cells"].items():
            w.writerow(dict(task=task, cell=k, status=c["status"], value=json.dumps(c["value"], default=str)[:200],
                            criterion=c["criterion"][:160], provenance=c["provenance"][:200], wall=c.get("wall")))
    return out


def run_task(task, model, dev, n_prompts, n_gauge, cells, seed=0, batch=32, smoke=False):
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    nd, nh = n_prompts
    d = BUILDERS[task](tok, nd, seed=100 + seed); dh = BUILDERS[task](tok, nh, seed=900 + seed)
    view = GPT2NodeView(model).to(dev).eval()
    tokens = torch.as_tensor(d["tokens"], device=dev); targets = make_targets(d, dev)
    ho = torch.as_tensor(dh["tokens"], device=dev); ho_t = make_targets(dh, dev)
    ctok = torch.as_tensor(d["corrupt"], device=dev)
    means = view.node_means(tokens, batch=batch)
    with torch.no_grad():
        m_clean = float(view.metric_per_example(tokens, targets, batch=batch).mean())
        m_corrupt = float(view.metric_per_example(ctok, targets, batch=batch).mean())
    prev = os.path.join(OUT, f"satono_crown_{task}.json")
    out = json.load(open(prev)) if os.path.exists(prev) else dict(task=task, model="gpt2-small", layer=LAYER, cells={})
    out.update(task=task, model="gpt2-small", layer=LAYER, smoke=bool(smoke),
               behaviour=dict(name=task, seq_len=d["seq_len"], n_discovery=nd, n_heldout=nh, n_nodes=view.n_nodes,
                              metric="logit difference at the last position (SPEC_O2 1)",
                              clean_logit_diff=m_clean, corrupted_logit_diff=m_corrupt,
                              meta={k: v for k, v in d.items() if k in ("n_ragged", "n_year_tokens", "dropped_year_tokens")}))
    C = out["cells"]; C.update(not_reached_cells())
    behavior = dict(name=task, kind="general", n=nd, seq_len=d["seq_len"])
    t0 = time.time()
    if "gauge" in cells: C["gauge"] = cell_gauge(model, view, d, dev, n_gauge); log(task, "gauge", C["gauge"])
    if "P0" in cells: C["P0"] = cell_P0(view, tokens, targets, means, seed=seed, batch=min(64, nd)); log(task, "P0", C["P0"])
    if "P2" in cells: C["P2"] = cell_P2(model, view, dev, tokens, targets, n_gauge, batch=min(batch, nd)); log(task, "P2", C["P2"])
    if "glue" in cells: C["glue"] = cell_glue(view, task, tokens, targets, means, seed=seed); log(task, "glue", C["glue"])
    if "P6" in cells: C["P6"] = cell_P6(view, tokens, seed=seed); log(task, "P6", C["P6"])
    if "P1" in cells: C["P1"] = cell_P1(model, view, d, dev, tokens, targets, means, n_gauge, batch=batch, seed=seed); log(task, "P1", C["P1"])
    p1 = C.get("P1", cell("not_reached", provenance="P1 not run in this invocation"))
    if "R3" in cells: C["R3"] = cell_R3(view, p1, ho, ho_t, means, batch=batch, seed=seed); log(task, "R3", C["R3"])
    if "R4" in cells: C["R4"] = cell_R4(view, p1, ho, means, batch=max(8, batch // 2), seed=seed); log(task, "R4", C["R4"])
    if "R5" in cells: C["R5"] = cell_R5(view, p1, ho, ho_t, means, batch=batch); log(task, "R5", C["R5"])
    if "R6" in cells: C["R6"] = cell_R6(view, p1, ho, ho_t, means, batch=batch); log(task, "R6", C["R6"])
    out["wall_total_s"] = round(time.time() - t0, 1)
    write(task, out, behavior)
    del view; torch.cuda.empty_cache()
    return out


def log(task, k, c):
    print(f"[{task}] {k:6s} {c['status']:12s} {json.dumps(c['value'], default=str)[:120]}  [{c.get('wall')}s]", flush=True)


def merge(tasks):
    """The merged T2 scorecard the O5 grid assembler consumes: one 'cells' dict and one 'scorecard'."""
    outs = {}
    for t in tasks:
        p = os.path.join(OUT, f"satono_crown_{t}.json")
        if os.path.exists(p): outs[t] = json.load(open(p))
    if not outs: return None
    allc = sorted({c for o in outs.values() for c in o["cells"]})
    cells = {}
    for c in allc:
        st = [o["cells"][c]["status"] for o in outs.values() if c in o["cells"]]
        if not st: continue
        agg = "certified" if all(s == "certified" for s in st) else \
              ("not_reached" if all(s == "not_reached" for s in st) else
               ("failed" if any(s == "failed" for s in st) else "partial"))
        cells[c] = cell(agg, value={t: outs[t]["cells"][c]["value"] for t in outs if c in outs[t]["cells"]},
                        criterion=next(o["cells"][c]["criterion"] for o in outs.values() if c in o["cells"]),
                        provenance="aggregate over the three T2 behaviours; 'certified' only if certified on all three (per-task detail in satono_crown_<task>.json)",
                        per_task={t: outs[t]["cells"][c]["status"] for t in outs if c in outs[t]["cells"]})
    gam = {t: (outs[t]["cells"].get("R4", {}).get("value") or {}).get("gamma_cov") for t in outs}
    gam = {t: v for t, v in gam.items() if v is not None}
    card = scorecard("gpt2-small (T2)", cells, dict(name="IOI / induction / greater-than", kind="general"))
    card.gamma_cov = float(np.median(list(gam.values()))) if gam else None
    merged = dict(model="gpt2-small", tier="T2", run="O2.1 Satono Crown", spec="experiments/O_opener/SPEC_O2.md",
                  tasks=list(outs), cells=cells, gamma_cov_per_task=gam, scorecard=card.to_dict())
    json.dump(merged, open(os.path.join(OUT, "satono_crown_gpt2.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "satono_crown_gpt2.md"), "w").write(card.to_markdown() + "\n")
    print(card.to_markdown())
    return merged


# ============================================================================================== O2.2
def audit():
    """O2.2 Narita Top Road: carry the F11 audit rows into the grid. No recompute (FINAL_SPEC 8.2 O2.2)."""
    import pandas as pd
    src = os.path.join(ROOT, "experiments", "F_gauge", "results", "F11_audit_table.csv")
    idf = os.path.join(ROOT, "experiments", "F_gauge", "results", "F11_identity.json")
    if not os.path.exists(src):
        out = dict(run="O2.2 Narita Top Road", cells=dict(audit=cell("not_reached", provenance=f"F11 results not on disk ({src})")))
        json.dump(out, open(os.path.join(OUT, "narita_top_road.json"), "w"), indent=1, default=str); print("not_reached: no F11 table"); return out
    df = pd.read_csv(src)
    rows = []
    for _, r in df.iterrows():
        va = r.get("verdict_aligned")
        rows.append(dict(quantity=r["metric"], n=int(r["n"]), verdict=r["verdict"],
                         verdict_aligned=(None if (isinstance(va, float) and np.isnan(va)) else va),
                         align=r.get("align"), layer=int(r["layer"]),
                         cv={k: (None if np.isnan(r[f"cv_{k}"]) else float(r[f"cv_{k}"])) for k in ("perm", "scale", "rot", "combined")}))
    def counts(key):
        c = {"invariant": 0, "covariant": 0, "artifact": 0}
        for x in rows:
            v = x[key]
            if v in c: c[v] += 1
        return c
    raw = counts("verdict"); aligned = counts("verdict_aligned")
    rescued = [x["quantity"] for x in rows if x["verdict"] == "artifact" and x["verdict_aligned"] == "invariant"]
    still = [x["quantity"] for x in rows if x["verdict"] == "artifact" and x["verdict_aligned"] in (None, "artifact")]
    ident = json.load(open(idf)) if os.path.exists(idf) else None
    val = dict(n_quantities=len(rows), raw=raw, aligned_where_defined=aligned,
               artifact_rescued_by_alignment=rescued, artifact_after_alignment=still)
    c = cell("certified", value=val,
             criterion="F11 classification per quantity (invariants.check: invariant iff CV < INVARIANT_CV over 30 renamings of kinds perm/scale/rot/combined; covariant iff permutation-invariant with a fitted power law R2 > 0.99; artifact otherwise). Carried unchanged: O2.2 recomputes nothing",
             provenance=f"experiments/F_gauge/results/F11_audit_table.csv (n=30, layer 6, GPT-2 small) + F11_identity.json; runner experiments/F_gauge/narita_brian.py (= run_F11.py)",
             rows=rows, identity_max_abs_dlogit=ident)
    out = dict(run="O2.2 Narita Top Road", model="gpt2-small", tier="T2", source=os.path.relpath(src, ROOT),
               note="carry, not a run: the F11 audit table is the O2.2 result (FINAL_SPEC 8.2)", cells=dict(audit=c))
    json.dump(out, open(os.path.join(OUT, "narita_top_road.json"), "w"), indent=1, default=str)
    print(json.dumps(val, indent=1))
    return out


# ============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--n-prompts", type=int, default=None, help="override both splits for every task")
    ap.add_argument("--n-gauge", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    if a.audit: audit(); return
    if a.merge_only: merge(a.tasks); return
    if a.smoke: a.n_gauge = min(a.n_gauge, 2)
    from transformers import GPT2LMHeadModel
    dev = a.device
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    cells = a.cells or CELLS
    for t in a.tasks:
        p = os.path.join(OUT, f"satono_crown_{t}.json")
        if os.path.exists(p) and not a.force and not a.cells:
            prev = json.load(open(p)); done = prev["cells"]
            # a --smoke JSON is a toy and must never make the full run look finished
            if prev.get("smoke") and not a.smoke: print("re-running", t, "(existing JSON is from --smoke)")
            elif all(c in done for c in cells): print("skip", t, "(done)"); continue
        n = (a.n_prompts, a.n_prompts) if a.n_prompts else (SMOKE_N[t] if a.smoke else NPROMPTS[t])
        print(f"== {t}: n_discovery={n[0]} n_heldout={n[1]} n_gauge={a.n_gauge} cells={cells}", flush=True)
        run_task(t, model, dev, n, a.n_gauge, cells, batch=a.batch, smoke=a.smoke)
    merge(a.tasks)
    audit()


if __name__ == "__main__":
    main()
