#!/usr/bin/env python3
"""O4.1 Bamboo Memory — T4 frontier open() (FINAL_SPEC §3, §5 T4 row, §6, §8.1, §8.2 O4.1, §12; pre-registered in SPEC_O4.md).

Two modes, one file:

  # 1. new compute (GPU, >= 40 GB for the frontier models) — the P0 gate of §3 step 2, the ONLY thing O4.1 computes itself
  python bamboo_memory.py --p0 --model meta-llama/Meta-Llama-3-8B --n-tokens 2000 --layers 14 15 16 17 18 \
      --out hpcc/results/o41 --dtype bfloat16 --resume

  # 2. assembly (CPU, no model) — §3 step 9: read the frontier jobs that are already written and report
  python bamboo_memory.py --assemble --model google/gemma-2-9b --results hpcc/results --out hpcc/results/o41

**This block composes; it does not re-implement.** The assembled cells come from jobs that already exist and carry their
own pre-registration: F12 `hpcc/kitasan_black.py` (step 1 gauge identity + steps 3/4 audit table, `symmetry.llama`
backend), C4 `hpcc/sirius_symboli.py` (P1 certified feature inventory), K8 `hpcc/biwa_hayahide.py` (Gemma Scope
reconstruction residual -> gamma_cov, step 7 / tool-side P4), L7 `hpcc/kawakami_princess.py` (end-to-end audit timing and
the `does_not_scale` list).  The P0 attribution graph reuses V7's EAP construction (`hpcc/mejiro_ardan.py`) with one
documented adaptation for sequential-residual blocks (Llama-3 / Gemma-2 / GPT-2 read the MLP after the attention of the
same layer, so there are two gradient read points per layer and within-layer head -> MLP edges); see SPEC_O4.md §"Step 2".

`--assemble` **never raises**: a missing or mismatched input is a `not_reached` cell carrying the reason and the command
that would produce it (Operating Constitution #4 — negatives are deliverables).  An unknown coverage gap is reported as
unknown, never as zero (§12.9: the word "complete" is not used anywhere in this block).

Outputs
  --p0        <out>/p0_<slug>.json         verdict, Q / treewidth / sparsity at densities 0.01, 0.05, 0.1, rewiring null
              <out>/p0_<slug>_E.npz        the signed EAP score matrix + node names + the resume state
  --assemble  <out>/scorecard_<slug>.json  Block-O layout: top-level "cells" (P0..P7 + gauge/audit/R4/L7) and "scorecard"
                                           (`whiteboxtopo.open.Scorecard.to_dict()` with gamma_cov) — what
                                           `experiments/O_opener/symboli_kris_s.py` (O5) consumes
              <out>/scorecard_<slug>.md    the human-readable scorecard, with the §12.9 "opened to degree" wording

Failure boundary: the P0 path handles the decoder-only families whose blocks it can name — Llama/Gemma-2/Qwen2/Mistral
(`model.model.layers`) and GPT-2 (`model.transformer.h`).  Anything else raises at start-up with the architecture name,
rather than silently scoring the wrong tensors.  Gemma-2's post-sublayer RMSNorms are folded into the per-node
contributions as a per-token rescale; the run records `gemma_sublayer_norm_split: true`, and the cell provenance says
that the split is exact as a decomposition of the realised contribution and first-order only as a counterfactual.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, platform, re, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "K_darkmatter"))

DEFAULT_RESULTS = os.path.join(ROOT, "hpcc", "results")
DEFAULT_OUT = os.path.join(DEFAULT_RESULTS, "o41")
CELLS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
DENSITIES = (0.01, 0.05, 0.1)
PREREG_DENSITY = 0.1            # SPEC_O4 step 2 / AMENDMENTS #5d of O1.2
TW_TRACTABLE, TW_MARGINAL = 8, 32
FLOOR_FACTOR = 10.0             # SPEC_O4 step 1: a kind is exact at this precision iff dev <= 10 x the perm floor
IDENTITY_FAIL = 0.5             # SPEC_O4 K-O4-1 (the L7 `does_not_scale` threshold)
INVARIANT_CV = 0.01             # whiteboxtopo.invariants.check.INVARIANT_CV
GAMMA_SUBSAMPLE, GAMMA_Q, GAMMA_SEED = 20000, 0.99, 0
V8_NOTE = ("the V-block phase diagram is the reference placement, but V8 found no calibrated structure->tractability map "
           "(whiteboxtopo.open.GATE_RULE; legibility.profile.V8_CALIBRATION_ATTEMPT), so this is a placement on the "
           "pre-registered FPT rule, not a calibrated prediction")


# ------------------------------------------------------------------------------------------------------ small helpers
def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model.split("/")[-1]).strip("_").lower()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def model_keys(model: str) -> list:
    """Normalised strings that a results directory or a recorded model field may carry for this model."""
    tail = model.split("/")[-1]; keys = [_norm(model), _norm(tail)]
    if _norm(tail).startswith("meta"): keys.append(_norm(tail)[4:])          # Meta-Llama-3-8B -> llama38b (dirs say f12_llama3_8b)
    return [k for k in dict.fromkeys(keys) if k]


def same_model(recorded, model) -> bool:
    if recorded is None: return True                                          # nothing recorded: the caller states the caveat
    r = _norm(recorded); return any(k in r or r in k for k in model_keys(model))


def cell(status, value=None, criterion="", provenance="", **evidence):
    """One scorecard cell in the Block-O convention (meisho_doto.py)."""
    return dict(status=status, value=value, criterion=criterion, provenance=provenance, evidence=evidence)


def _j(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, dict): return {str(k): _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(v) for v in o]
    return o


# =========================================================================================== P0: the frontier EAP graph
class Arch:
    """The three node/read points this job needs, named per family.  Raises on an architecture it cannot name."""

    def __init__(self, model):
        import torch
        name = type(model).__name__; cfg = model.config
        if hasattr(model, "model") and hasattr(getattr(model, "model"), "layers"):
            self.family = "gemma2" if name.startswith("Gemma2") else "llama"
            self.blocks = model.model.layers; self.conv1d = False
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.family = "gpt2"; self.blocks = model.transformer.h; self.conv1d = True
        else:
            raise NotImplementedError(f"bamboo_memory --p0 does not know how to name the blocks of {name}; "
                                      "supported: Llama/Gemma-2/Qwen2/Mistral (model.model.layers) and GPT-2 (model.transformer.h)")
        self.name = name; self.cfg = cfg
        self.H = int(cfg.num_attention_heads); self.d = int(cfg.hidden_size); self.n_layers = int(cfg.num_hidden_layers)
        self.dh = int(getattr(cfg, "head_dim", None) or (self.d // self.H))
        if self.conv1d: self.dh = self.d // self.H
        self.norm_split = self.family == "gemma2"

    def o_proj(self, l): return self.blocks[l].attn.c_proj if self.conv1d else self.blocks[l].self_attn.o_proj

    def attn_prenorm(self, l):
        """The norm whose single consumer is the attention QKV projection (the per-head read point of #O4.1 (d))."""
        b = self.blocks[l]; return b.ln_1 if self.conv1d else b.input_layernorm

    def qkv_mods(self, l):
        """[(attribute name, module, mode)] for the QKV projection(s). ``mode`` says how a head's parameters sit in the
        weight: ``cols3`` = one Conv1D (d, 3d) with [q|k|v] blocks (GPT-2), ``rows`` = separate Linears (Llama family)."""
        b = self.blocks[l]
        if self.conv1d: return [("c_attn", b.attn.c_attn, "cols3")]
        a = b.self_attn
        return [("q_proj", a.q_proj, "rows"), ("k_proj", a.k_proj, "rows_kv"), ("v_proj", a.v_proj, "rows_kv")]

    def n_kv(self):
        return int(getattr(self.cfg, "num_key_value_heads", None) or self.H)

    def mlp(self, l): return self.blocks[l].mlp

    def mlp_prenorm(self, l):
        b = self.blocks[l]
        if self.family == "gemma2": return b.pre_feedforward_layernorm
        return b.ln_2 if self.conv1d else b.post_attention_layernorm

    def post_attn_norm(self, l): return self.blocks[l].post_attention_layernorm if self.family == "gemma2" else None

    def post_ffn_norm(self, l): return self.blocks[l].post_feedforward_layernorm if self.family == "gemma2" else None

    def mid_window(self):
        mid = self.n_layers // 2; return [l for l in range(mid - 2, mid + 3) if 0 <= l < self.n_layers]

    def describe(self):
        return dict(arch=self.name, family=self.family, n_layers=self.n_layers, H=self.H, d_head=self.dh,
                    d_model=self.d, gemma_sublayer_norm_split=self.norm_split)


_PERHEAD = {}


def _per_head_proj_cls():
    """``torch`` is imported lazily everywhere in this file (the --assemble path must run without it), so the
    per-head projection module is defined on first use."""
    import torch
    if "cls" in _PERHEAD: return _PERHEAD["cls"]

    class _PerHeadProj(torch.nn.Module):
        """Recompute one QKV projection head by head, each head reading its OWN clone of the block input through the
        block's attention pre-norm, so dL/dx_in is head-specific (O4.1 AMENDMENTS #O4.1 (d); the same repair as
        V AMENDMENTS #15 / SPEC_O3 #O3-2). Numerically identical to the original projection — the per-head parameter
        blocks are disjoint and are written back into their own slots — at the same matmul flops (the pre-norm is
        recomputed H times, which is negligible).

        ``mode``: ``cols3`` a single Conv1D of shape (d, 3d) whose columns are [q|k|v] with head h at ``j*d + h*dh``
        (GPT-2); ``rows`` a Linear whose rows ``h*dh:(h+1)*dh`` are head h (Llama q_proj); ``rows_kv`` the same but
        indexed by the KV head ``h // group`` — under GQA the K/V rows are SHARED by a query group, so they are computed
        from the clone of the group's FIRST query head (the representative), which is recorded in the cell provenance."""

        def __init__(self, mod, ln, arch, l, caps, mode):
            super().__init__(); self.mod, self.ln, self.arch, self.l, self.caps, self.mode = mod, ln, arch, l, caps, mode

        def _eps(self, x, h):
            """head h's zero leaf on the block input, created once per forward and shared by q/k/v."""
            key = (self.l, f"h{h}")
            e = self.caps.get(key)
            if e is None or e.shape != x.shape or not e.requires_grad:
                e = torch.zeros_like(x, requires_grad=True); self.caps[key] = e
            return e

        def forward(self, x_ln):                      # x_ln = prenorm(x); ignored, recomputed per head
            A = self.arch; H, dh, d = A.H, A.dh, A.d
            x = self.caps[(self.l, "_x")]
            W = self.mod.weight; b = getattr(self.mod, "bias", None)
            if self.mode == "cols3":                                          # Conv1D: out = x @ W + b, W (d, 3d)
                idx = torch.as_tensor(np.concatenate([[j * d + h * dh + i for j in range(3) for i in range(dh)]
                                                      for h in range(H)]), device=W.device)
                parts = []
                for h in range(H):
                    cols = idx[h * 3 * dh:(h + 1) * 3 * dh]
                    y = self.ln(x + self._eps(x, h)) @ W.index_select(1, cols)
                    if b is not None: y = y + b.index_select(0, cols)
                    parts.append(y)
                cat = torch.cat(parts, -1)
                return torch.zeros(x.shape[:-1] + (3 * d,), dtype=cat.dtype, device=cat.device).index_copy(-1, idx, cat)
            n_kv = A.n_kv(); group = max(H // max(n_kv, 1), 1)
            n_out = H if self.mode == "rows" else n_kv
            rows_per = W.shape[0] // max(n_out, 1)
            parts = []
            for j in range(n_out):
                h = j if self.mode == "rows" else j * group                   # GQA: the group's FIRST query head
                r = slice(j * rows_per, (j + 1) * rows_per)
                y = torch.nn.functional.linear(self.ln(x + self._eps(x, h)), W[r], None if b is None else b[r])
                parts.append(y)
            return torch.cat(parts, -1)

    _PERHEAD["cls"] = _PerHeadProj
    return _PerHeadProj


def _install_per_node(model, arch, layers, caps):
    """#O4.1 (d): per-TARGET-NODE read points. Every head gets its own clone of the block input (through a temporary
    wrapper on the QKV projection) and the MLP gets its own clone at its pre-norm, so dL/dx_in(v) is node-specific
    instead of one gradient per read point. Returns (handles, restore) — `restore` must be called in a finally."""
    import torch
    cls = _per_head_proj_cls(); hs = []; orig = []

    def pre_keep_x(l):
        def f(mod, inp): caps[(l, "_x")] = inp[0]          # kept in the graph; MUST return None
        return f

    def pre_mlp(l):
        def f(mod, inp):
            e = torch.zeros_like(inp[0], requires_grad=True); caps[(l, "mlp")] = e
            return (inp[0] + e,) + tuple(inp[1:])          # a pre-hook's return REPLACES the inputs
        return f

    def pre_det(key):
        def f(mod, inp): caps[key] = inp[0].detach()
        return f

    def post_det(key):
        def f(mod, inp, out): caps[key] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    for l in layers:
        hs.append(arch.blocks[l].register_forward_pre_hook(pre_keep_x(l)))
        hs.append(arch.mlp_prenorm(l).register_forward_pre_hook(pre_mlp(l)))
        hs.append(arch.o_proj(l).register_forward_pre_hook(pre_det((l, "z"))))
        hs.append(arch.o_proj(l).register_forward_hook(post_det((l, "attn_out"))))
        hs.append(arch.mlp(l).register_forward_hook(post_det((l, "mlp_out"))))
        parent = arch.blocks[l].attn if arch.conv1d else arch.blocks[l].self_attn
        for nm, mod, mode in arch.qkv_mods(l):
            orig.append((parent, nm, mod))
            setattr(parent, nm, cls(mod, arch.attn_prenorm(l), arch, l, caps, mode))

    def restore():
        for h_ in hs: h_.remove()
        for parent, nm, mod in orig: setattr(parent, nm, mod)
    return hs, restore


def _install(model, arch, layers, caps):
    """Forward hooks capturing, per layer: the two gradient read points (block input, pre-MLP-norm input — both kept in
    the autograd graph) and the node activations (o_proj input z, o_proj output, MLP output — detached)."""
    hs = []

    def pre_keep(key):
        def f(mod, inp):
            caps[key] = inp[0]                      # MUST return None: a pre-hook's return value replaces the inputs
        return f

    def pre_det(key):
        def f(mod, inp): caps[key] = inp[0].detach()
        return f

    def post_det(key):
        def f(mod, inp, out): caps[key] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    for l in layers:
        hs.append(arch.blocks[l].register_forward_pre_hook(pre_keep((l, "attn_in"))))
        hs.append(arch.mlp_prenorm(l).register_forward_pre_hook(pre_keep((l, "mlp_in"))))
        hs.append(arch.o_proj(l).register_forward_pre_hook(pre_det((l, "z"))))
        hs.append(arch.o_proj(l).register_forward_hook(post_det((l, "attn_out"))))
        hs.append(arch.mlp(l).register_forward_hook(post_det((l, "mlp_out"))))
    return hs


def _iter_contribs(arch, caps, layers):
    """Yield (node, contribution) one node at a time — the node's own additive contribution to the residual stream,
    (B, T, d) float32.  Heads: z_h @ W_O[:, h-slice]^T.  MLP: the MLP module output.  For Gemma-2 both are rescaled by
    the factor (1 + w) / rms(sublayer_out) of the post-sublayer RMSNorm, which makes the split an EXACT decomposition of
    the realised residual contribution (verified: sum of nodes = h_{l+1} - h_l to 3e-7 relative on a random Gemma-2
    config); it is first-order only as a *counterfactual*, since ablating one node would also move rms(sublayer_out)."""
    import torch
    for l in layers:
        z = caps[(l, "z")].float(); W = arch.o_proj(l).weight.float()
        fa = fm = None
        if arch.norm_split:
            nrm = arch.post_attn_norm(l); ao = caps[(l, "attn_out")].float()
            fa = (1.0 + nrm.weight.float()).view(1, 1, -1) * torch.rsqrt(ao.pow(2).mean(-1, keepdim=True) + nrm.eps)
            nrm2 = arch.post_ffn_norm(l); mo_ = caps[(l, "mlp_out")].float()
            fm = (1.0 + nrm2.weight.float()).view(1, 1, -1) * torch.rsqrt(mo_.pow(2).mean(-1, keepdim=True) + nrm2.eps)
        for h in range(arch.H):
            sl = slice(h * arch.dh, (h + 1) * arch.dh)
            c = z[..., sl] @ (W[sl, :] if arch.conv1d else W[:, sl].T)
            yield (l, f"h{h}"), (c * fa if fa is not None else c)
        mo = caps[(l, "mlp_out")].float()
        yield (l, "mlp"), (mo * fm if fm is not None else mo)


def eap_matrix(model, arch, T, layers, batch, out_npz, resume=False, verbose=True, per_node=False):
    """Signed EAP score matrix E[u, v] = sum_t (mean_u - a_u(t)) . dL/dx_in(v)(t), L = summed next-token log-probability,
    mean corruption over the whole probe (V7 `mejiro_ardan.eap_graph`, adapted to sequential-residual blocks: two read
    points per layer).  Resumable per batch through ``out_npz``."""
    import torch
    nodes = [(l, f"h{h}") for l in layers for h in range(arch.H)] + [(l, "mlp") for l in layers]
    nidx = {n: i for i, n in enumerate(nodes)}
    if per_node:                       # #O4.1 (d): one read point per TARGET NODE (the V AMENDMENTS #15 repair)
        reads = list(nodes); read_of = {n: n for n in nodes}
    else:                              # PRE-REGISTERED: two read points per layer (block input, pre-MLP-norm input)
        reads = [(l, "attn_in") for l in layers] + [(l, "mlp_in") for l in layers]
        read_of = {n: ((n[0], "mlp_in") if n[1] == "mlp" else (n[0], "attn_in")) for n in nodes}
    ridx = {r: i for i, r in enumerate(reads)}
    allow = np.zeros((len(nodes), len(nodes)), bool)
    for u in nodes:
        for v in nodes:
            if u[0] < v[0] or (u[0] == v[0] and u[1] != "mlp" and v[1] == "mlp"): allow[nidx[u], nidx[v]] = True
    for p in model.parameters(): p.requires_grad_(False)
    dev = next(model.parameters()).device; d = arch.d; nb = int(np.ceil(len(T) / batch))

    state = None
    if resume and os.path.exists(out_npz):
        z = np.load(out_npz, allow_pickle=True)
        if list(z["node_names"]) == [f"{l}.{k}" for l, k in nodes]:
            state = dict(E=z["E"], means=z["means"], n_tok=int(z["n_tok"]), done=int(z["done_batches"]))
            if verbose: print(f"[p0] resume: {state['done']}/{nb} batches done", flush=True)

    # ---- pass 1: node means over the whole probe (mean corruption), no grad
    if state is not None:
        means_np = state["means"]
    else:
        sums = torch.zeros(len(nodes), d, dtype=torch.float64, device=dev); n_tok = 0
        for b in range(0, len(T), batch):
            ids = T[b:b + batch]; caps = {}; hs = _install(model, arch, layers, caps)
            with torch.no_grad(): model(input_ids=ids)
            for h_ in hs: h_.remove()
            for u, c in _iter_contribs(arch, caps, layers): sums[nidx[u]] += c.reshape(-1, d).double().sum(0)
            n_tok += int(ids.shape[0] * ids.shape[1]); caps.clear()
            if verbose: print(f"[p0] means {b // batch + 1}/{nb}", flush=True)
        means_np = (sums / max(n_tok, 1)).cpu().numpy()
        state = dict(E=np.zeros((len(nodes), len(nodes))), means=means_np, n_tok=n_tok, done=0)
    means = torch.as_tensor(state["means"], dtype=torch.float32, device=dev)

    # ---- pass 2: gradients at the read points, scores accumulated
    E = np.asarray(state["E"], dtype=np.float64).copy(); done = state["done"]
    emb_layer = model.get_input_embeddings()
    for bi in range(done, nb):
        t0 = time.time(); ids = T[bi * batch:(bi + 1) * batch]
        emb = emb_layer(ids).detach().requires_grad_(True)
        caps = {}
        if per_node: hs, restore = _install_per_node(model, arch, layers, caps)
        else: hs = _install(model, arch, layers, caps); restore = lambda: [h_.remove() for h_ in hs]
        try:
            logits = model(inputs_embeds=emb).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1).sum()
            grads = torch.autograd.grad(lp, [caps[r] for r in reads], retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(emb[..., :1].expand(*emb.shape)) for g in grads]
        finally:
            restore()
        G = torch.stack([g.float() for g in grads])                                   # (R, B, T, d)
        del logits, lp, grads
        for u, c in _iter_contribs(arch, caps, layers):
            s = torch.einsum("btd,rbtd->r", means[nidx[u]].view(1, 1, d) - c, G).double().cpu().numpy()
            row = nidx[u]
            for v in nodes:
                if allow[row, nidx[v]]: E[row, nidx[v]] += s[ridx[read_of[v]]]
        del G, emb, caps
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        np.savez(out_npz, E=E, means=state["means"], n_tok=state["n_tok"], done_batches=bi + 1, per_node=int(per_node),
                 node_names=np.array([f"{l}.{k}" for l, k in nodes]), layers=np.array(layers))
        if verbose: print(f"[p0] eap {bi + 1}/{nb} ({time.time() - t0:.1f}s)", flush=True)
    return E, [f"{l}.{k}" for l, k in nodes]


def p0_profiles(E, densities=DENSITIES, seed=0, n_null=20):
    """(Q vs the degree-preserving rewiring null, min-fill treewidth bound, sparsity) at each edge density."""
    from whiteboxtopo import graphs
    from whiteboxtopo.legibility.graph import sparsity as node_sparsity
    A_full = graphs.symmetrise(E); imp = np.abs(np.asarray(E)).sum(1); out = {}
    for dens in densities:
        A = graphs.threshold_density(A_full, dens)
        Q, memb = graphs.modularity(A, seed=seed)
        q0, sd = graphs.rewired_null(A, n=n_null, seed=seed)
        out[f"{dens}"] = dict(density=float(dens), n_edges=int((A != 0).sum() // 2), Q=float(Q),
                              Q_null_rewired_mean=float(q0), Q_null_rewired_sd=float(sd), Q_excess=float(Q - q0),
                              n_communities=int(len(np.unique(memb))), treewidth_ub=int(graphs.treewidth_ub(A)),
                              sparsity_node_importance=float(node_sparsity(imp)), sparsity_edges=float(graphs.sparsity(A_full)))
    return out


def p0_gauge_orbit(model, arch, T, layers, a, E_pre, E_pn, names, n_gauge=None):
    """A small K-O-2 orbit on BOTH graphs when `whiteboxtopo.symmetry` has a backend for this model (symmetry.llama
    covers Llama/Gemma-2/Qwen2/Mistral; symmetry.gpt2 covers GPT-2; symmetry.neox covers Pythia). Reports the CV of Q,
    the treewidth bound and sparsity at the pre-registered density. Absent backend -> a recorded reason, not an error."""
    n_gauge = a.n_gauge if n_gauge is None else n_gauge
    if n_gauge <= 0:
        return dict(ran=False, reason="--n-gauge 0")
    import torch
    dt = next(model.parameters()).dtype
    if dt not in (torch.float32, torch.float64):
        return dict(ran=False, reason=f"model dtype is {dt}: whiteboxtopo.symmetry.{{gpt2,llama,neox}}.apply inverts the "
                                      f"per-head basis change with torch.linalg.inv, which refuses low-precision dtypes, "
                                      f"and a K-O-2 CV measured at the bf16 floor (~1e-2) could not resolve ART_CV = 1e-3 "
                                      f"anyway. Re-run with --dtype float32 for the orbit; the P0 verdict is unaffected.")
    # Cost bound (stated, not tuned): the orbit measures gauge SENSITIVITY, not the reported numbers, so it runs on a
    # truncated probe -- the first `--gauge-seq` sequences. 0 = the full probe. Recorded in the result.
    Tg = T if not a.gauge_seq else T[:max(1, min(int(a.gauge_seq), len(T)))]
    from whiteboxtopo import symmetry as S, graphs
    try:
        be = S._backend(model)
    except Exception as e:
        return dict(ran=False, reason=f"no symmetry backend for {type(model).__name__}: {e}")

    def prof(M):
        A_full = graphs.symmetrise(M); A = graphs.threshold_density(A_full, PREREG_DENSITY)
        Q, memb = graphs.modularity(A, seed=a.seed)
        return dict(Q=float(Q), tw=float(graphs.treewidth_ub(A)), sp=float(graphs.sparsity(A_full)),
                    cs=tuple(sorted(np.bincount(np.asarray(memb)).tolist(), reverse=True)))

    base = {}
    for lbl, pnflag in (("pre_registered", False), ("per_node", True)):
        if os.path.exists(tmp0 := os.path.join(a.out, "_orbit_base.npz")): os.remove(tmp0)
        M, _ = eap_matrix(model, Arch(model), Tg, layers, a.batch, tmp0, resume=False, verbose=False, per_node=pnflag)
        base[lbl] = prof(M)
    if os.path.exists(os.path.join(a.out, "_orbit_base.npz")): os.remove(os.path.join(a.out, "_orbit_base.npz"))
    vals = {lbl: [base[lbl]] for lbl in base}
    devs = []
    rng = np.random.default_rng(0)
    tmp = os.path.join(a.out, "_orbit_tmp.npz")
    for j in range(n_gauge):
        g = be.sample(model, rng, kind="combined", orthogonal=False)
        gm = S.apply(model, g)
        try:
            with torch.no_grad():
                y0 = model(T[:1, :32]).logits.float(); y1 = gm(T[:1, :32]).logits.float()
            devs.append(float((y1 - y0).abs().max() / max(float(y0.abs().mean()), 1e-12)))
            for lbl, pnflag in (("pre_registered", False), ("per_node", True)):
                if os.path.exists(tmp): os.remove(tmp)
                M, _ = eap_matrix(gm, Arch(gm), Tg, layers, a.batch, tmp, resume=False, verbose=False, per_node=pnflag)
                vals[lbl].append(prof(M))
        finally:
            del gm
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    if os.path.exists(tmp): os.remove(tmp)

    def cv(v, k):
        x = np.array([d[k] for d in v], float); return float(x.std() / (abs(x.mean()) + 1e-12))
    out = dict(ran=True, n_gauge=int(n_gauge), backend=be.__name__, kind="combined", orthogonal=False,
               art_cv=1e-3, fp32_floor_rel_max=(max(devs) if devs else None),
               orbit_probe=dict(n_seq=int(len(Tg)), n_tokens=int(Tg.numel()),
                                truncated=bool(a.gauge_seq and len(Tg) < len(T))),
               note="SPEC_O K-O-2: CV < 1e-3 (or an identical set) is certifiable. The P0 verdict is the TREEWIDTH "
                    "bound at the pre-registered density, so `tw` is the quantity that matters here; Q is reported.")
    for lbl, v in vals.items():
        out[lbl] = dict(Q_cv=cv(v, "Q"), treewidth_cv=cv(v, "tw"), sparsity_cv=cv(v, "sp"),
                        community_sizes_identical=bool(all(d["cs"] == v[0]["cs"] for d in v)),
                        Q=[d["Q"] for d in v], treewidth=[d["tw"] for d in v])
    return out


def verdict_from_tw(tw: int) -> str:
    return "tractable" if tw <= TW_TRACTABLE else ("marginal" if tw <= TW_MARGINAL else "intractable")


def load_probe(tok, n_tokens, seq_len, cache_dir):
    """The fixed OpenWebText-10k probe every K/V/F job uses (BOS + seq_len-1 tokens per sequence)."""
    os.makedirs(cache_dir, exist_ok=True)
    n_seq = max(1, int(np.ceil(n_tokens / (seq_len - 1))))
    p = os.path.join(cache_dir, "tokens.npy")
    if os.path.exists(p) and np.load(p).shape[0] < n_seq: os.remove(p)
    try:
        import eishin_flash
        eishin_flash.N_SEQ, eishin_flash.SEQ_LEN, eishin_flash.CACHE = n_seq, seq_len, cache_dir
        toks = eishin_flash.load_tokens(tok)
    except Exception as e:                      # the cluster may not have K1's hard-coded parquet path: same recipe, hub copy
        print(f"[p0] eishin_flash.load_tokens unavailable ({type(e).__name__}: {e}); rebuilding the same probe from the hub", flush=True)
        from datasets import load_dataset
        ds = load_dataset("stas/openwebtext-10k", split="train"); bos = tok.bos_token_id
        bos = tok.eos_token_id if bos is None else bos
        seqs = []
        for t in ds["text"]:
            ids = tok(t)["input_ids"]
            for s in range(0, len(ids) - (seq_len - 1) + 1, seq_len - 1):
                seqs.append([bos] + ids[s:s + seq_len - 1])
                if len(seqs) >= n_seq: break
            if len(seqs) >= n_seq: break
        toks = np.array(seqs[:n_seq], dtype=np.int64); np.save(p, toks)
    return np.asarray(toks)[:n_seq, :seq_len]


def run_p0(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True); sl = slug(a.model); t_all = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt).to(a.device).eval()
    arch = Arch(model)
    layers = a.layers if a.layers else arch.mid_window()
    bad = [l for l in layers if not 0 <= l < arch.n_layers]
    if bad: raise SystemExit(f"--layers {bad} outside 0..{arch.n_layers - 1} for {a.model}")
    toks = load_probe(tok, a.n_tokens, a.seq_len, os.path.join(a.out, f"probe_{sl}"))
    T = torch.as_tensor(toks, dtype=torch.long, device=a.device)
    n_tok = int(T.numel())
    print(f"[p0] {a.model} {arch.describe()} layers={layers} probe={tuple(T.shape)} ({n_tok} tokens) dtype={a.dtype}", flush=True)
    t0 = time.time()
    E, names = eap_matrix(model, arch, T, layers, a.batch, os.path.join(a.out, f"p0_{sl}_E.npz"), resume=a.resume)
    t_eap = time.time() - t0
    # #O4.1 (d): the same graph with per-TARGET-NODE gradients (V AMENDMENTS #15 / SPEC_O3 #O3-2). Reported alongside;
    # the P0 VERDICT stays on the pre-registered graph. A failure here is recorded, never fatal.
    pn = dict(ran=False)
    if not a.no_per_node:
        t1 = time.time()
        try:
            Epn, _ = eap_matrix(model, arch, T, layers, a.batch, os.path.join(a.out, f"p0_{sl}_E_pernode.npz"),
                                resume=a.resume, per_node=True)
            pn = dict(ran=True, profiles=p0_profiles(Epn, seed=a.seed, n_null=a.n_null), wall_s=round(time.time() - t1, 1))
            pn["verdict"] = verdict_from_tw(pn["profiles"][f"{PREREG_DENSITY}"]["treewidth_ub"])
            for lbl, M in (("pre_registered", E), ("per_node", Epn)):
                v = np.abs(np.asarray(M) + np.asarray(M).T)[np.triu_indices(len(names), 1)]
                pn[f"n_distinct_edge_values_{lbl}"] = int(len(np.unique(v)))
            try:        # the orbit is a separate failure domain: it must not cost the per-node graph
                pn["gauge_orbit"] = p0_gauge_orbit(model, arch, T, layers, a, E, Epn, names)
            except Exception as e2:
                pn["gauge_orbit"] = dict(ran=False, reason=f"{type(e2).__name__}: {e2}",
                                         traceback=traceback.format_exc()[-600:])
                print(f"[p0] gauge orbit FAILED (recorded, not fatal): {pn['gauge_orbit']['reason']}", flush=True)
        except Exception as e:
            pn = dict(ran=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-800:])
            print(f"[p0] per-node graph FAILED (recorded, not fatal): {pn['error']}", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    t0 = time.time(); prof = p0_profiles(E, seed=a.seed, n_null=a.n_null); t_graph = time.time() - t0
    pr = prof[f"{PREREG_DENSITY}"]; verdict = verdict_from_tw(pr["treewidth_ub"])
    out = dict(job="O4.1 Bamboo Memory --p0", spec="experiments/O_opener/SPEC_O4.md (step 2 / P0)", model=a.model, slug=sl,
               **arch.describe(), layers=[int(l) for l in layers], nodes=names, n_nodes=len(names),
               n_tokens=n_tok, n_seq=int(T.shape[0]), seq_len=int(T.shape[1]), dtype=a.dtype, batch=a.batch,
               prereg_density=PREREG_DENSITY, profiles=prof, verdict=verdict, per_node=pn,
               criterion=(f"tractable iff treewidth_ub <= {TW_TRACTABLE} at density {PREREG_DENSITY}, marginal <= {TW_MARGINAL}, "
                          "else intractable (the FPT parameter, FINAL_SPEC 2.14; AMENDMENTS #5d of O1.2). Q is REPORTED against the "
                          "degree-preserving rewiring null and is not part of the verdict"),
               provenance=("EAP attribution graph over attention heads + MLP of the given layers, mean corruption, metric = summed "
                           "next-token log-probability; construction from hpcc/mejiro_ardan.py (V7) with two gradient read points per "
                           "layer for sequential-residual blocks (block input for the attention nodes, pre-MLP-norm input for the MLP "
                           "node), so within-layer head->MLP edges are scored; measures whiteboxtopo.graphs / legibility.graph; "
                           "null = whiteboxtopo.graphs.rewired_null(n=%d); probe = the fixed OpenWebText-10k slice used by K1/V7. %s"
                           % (a.n_null, V8_NOTE)),
               gemma_note=("Gemma-2 applies post_attention_layernorm / post_feedforward_layernorm to the sublayer output before the "
                           "residual add, so per-node contributions were rescaled by (1+w)/rms(sublayer_out). The split is an exact "
                           "decomposition of the realised contribution (sum of nodes = h_{l+1} - h_l, checked to 3e-7 relative on a "
                           "random Gemma-2 config) and first-order only as a counterfactual, since ablating a node would also move "
                           "rms(sublayer_out) (SPEC_O4 step 2)") if arch.norm_split else None,
               wall_s=dict(eap=round(t_eap, 1), graph=round(t_graph, 1), total=round(time.time() - t_all, 1)),
               host=platform.node(), date=time.strftime("%Y-%m-%d %H:%M"))
    p = os.path.join(a.out, f"p0_{sl}.json"); json.dump(_j(out), open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("model", "layers", "n_nodes", "n_tokens", "verdict", "wall_s")}, indent=1))
    print(f"[p0] density {PREREG_DENSITY}: Q {pr['Q']:.4f} (rewired null {pr['Q_null_rewired_mean']:.4f} +- "
          f"{pr['Q_null_rewired_sd']:.4f}), treewidth_ub {pr['treewidth_ub']}, sparsity {pr['sparsity_node_importance']:.4f} "
          f"-> {verdict}\n[p0] wrote {p}", flush=True)
    return out


# ============================================================================================ assemble: find the inputs
KNOWN_MODELS = ("meta-llama/Meta-Llama-3-8B", "google/gemma-2-9b", "google/gemma-2-2b",
                "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "gpt2")


def find_input(results, prefix, model, must_have):
    """Locate the results directory of a frontier job for THIS model.  Returns (path, note) or (None, why).

    K-O4-4: a directory is never borrowed across models.  A directory whose name identifies a *different* known model is
    skipped outright (F12's ``identity.json`` records no model field, so the directory name is the only evidence there);
    a directory that names no model at all is accepted with a stated caveat, which the caller puts in the provenance."""
    cands = sorted(d for d in glob.glob(os.path.join(results, prefix + "*")) if os.path.isdir(d))
    cands = [d for d in cands if all(glob.glob(os.path.join(d, m)) for m in must_have)]
    if not cands: return None, f"no directory {os.path.relpath(results, ROOT)}/{prefix}* containing {', '.join(must_have)}"
    keys = model_keys(model); named, anonymous, foreign = [], [], []
    for d in cands:
        nd = _norm(os.path.basename(d))
        if any(k in nd for k in keys): named.append(d)
        elif any(any(k in nd for k in model_keys(other)) for other in KNOWN_MODELS if other != model): foreign.append(d)
        else: anonymous.append(d)
    pool = named or anonymous
    if not pool:
        return None, (f"the only {prefix}* candidates name other models ({[os.path.basename(d) for d in foreign]}), not "
                      f"{model} — K-O4-4: results are never borrowed across models")
    pick = pool[0]; note = ""
    if not named:
        note = f"directory name {os.path.basename(pick)} does not name {model}; matched by content/position only"
    if len(pool) > 1:
        note += f" (several candidates {[os.path.basename(d) for d in pool]}; used the first)"
    return pick, note


def _read_json(p):
    try: return json.load(open(p))
    except Exception: return None


def _read_csv(p):
    try:
        with open(p) as f: return list(csv.DictReader(f))
    except Exception: return []


def gauge_cell(results, model):
    d, note = find_input(results, "f12", model, ["identity.json"]); why = note
    cmd = f"sbatch hpcc/sbatch/o41_frontier_p0.sbatch's F12 sibling: python hpcc/kitasan_black.py --model {model} --out hpcc/results/f12_{slug(model)} --resume"
    if d is None:
        return cell("not_reached", criterion="SPEC_O4 step 1", provenance=f"not reached: {why}. Run: {cmd}"), None, ""
    j = _read_json(os.path.join(d, "identity.json")) or {}
    ident = {k: float(v) for k, v in (j.get("identity") or {}).items()}
    if not ident:
        return cell("not_reached", criterion="SPEC_O4 step 1", provenance=f"not reached: {d}/identity.json has no identity block. Run: {cmd}"), None, ""
    floor = ident.get("perm")
    kinds = {k: ("exact_at_this_precision" if (floor is not None and v <= FLOOR_FACTOR * floor) else "approximate_in_" + str(j.get("dtype", "?")))
             for k, v in ident.items()}
    worst = max(ident.values())
    status = "failed" if worst > IDENTITY_FAIL else ("certified" if all(v == "exact_at_this_precision" for v in kinds.values()) else "partial")
    prov = (f"F12 hpcc/kitasan_black.py identity check, whiteboxtopo.symmetry.llama backend (mlp perms, KV/Q head perms, V/O "
            f"rotations, RoPE-commuting Q/K per-plane phases+scales, RMSNorm rescales), dtype {j.get('dtype')}, "
            f"{j.get('n_tokens')} tokens; source {os.path.relpath(d, ROOT)}/identity.json. "
            f"Floor rule (SPEC_O4 step 1): permutations reorder sums but change no value, so the perm deviation IS the "
            f"floating-point floor ({floor}); a kind is exact at this precision iff its deviation <= {FLOOR_FACTOR:g} x the floor"
            + (f". CAVEAT: {note}" if note else ""))
    if status == "failed": prov += f". K-O4-1 FIRED: worst deviation {worst:.3g} > {IDENTITY_FAIL}; certification at this precision is not usable — rerun with --dtype float32"
    return cell(status, value=dict(max_abs_dlogit=ident, float_floor=floor, kinds=kinds, dtype=j.get("dtype")),
                criterion=f"exact at this precision iff max|dlogit| <= {FLOOR_FACTOR:g} x the perm floor; the gauge step fails iff any kind > {IDENTITY_FAIL}",
                provenance=prov), d, note


def audit_cell(results, model, f12_dir, note=""):
    if f12_dir is None:
        return cell("not_reached", criterion="SPEC_O4 steps 3-4", provenance=f"not reached: no F12 run for {model} (see the gauge cell)")
    p = os.path.join(f12_dir, "F12_audit_table.csv"); rows = _read_csv(p)
    if not rows:
        return cell("not_reached", criterion="SPEC_O4 steps 3-4",
                    provenance=f"not reached: {os.path.relpath(p, ROOT)} absent or empty (F12 writes it after the first layer finishes)")
    table = {}
    for r in rows:
        if r.get("kind") != "combined": continue
        try: cv = float(r["median_cv"])
        except Exception: continue
        try: cva = float(r.get("median_cv_aligned", "nan"))
        except Exception: cva = float("nan")
        v = "invariant" if cv < INVARIANT_CV else ("covariant_aligned" if cva == cva and cva < INVARIANT_CV else "artifact")
        table.setdefault(v, []).append(f"L{r.get('layer')}:{r.get('metric')}")
    counts = {k: len(v) for k, v in sorted(table.items())}
    n = sum(counts.values())
    return cell("certified" if n else "not_reached", value=dict(counts=counts, n_quantities=n, by_class=table),
                criterion=(f"whiteboxtopo.invariants.check rule, verbatim (INVARIANT_CV = {INVARIANT_CV}) on the combined renaming: "
                           "invariant iff median_cv < 0.01; covariant (aligned) iff the unit-axis-aligned median_cv < 0.01; else artifact "
                           "(discarded, FINAL_SPEC 3.4). No T4 pass threshold exists in FINAL_SPEC 6 — this table is a deliverable (10.3), "
                           "reported, not claimed"),
                provenance=f"F12 hpcc/kitasan_black.py; full per-metric table at {os.path.relpath(p, ROOT)}" + (f". CAVEAT: {note}" if note else ""))


def p1_cell(results, model):
    d, note = find_input(results, "c4", model, ["C4_summary.json"]); why = note
    cmd = f"python hpcc/sirius_symboli.py --model {model} --out hpcc/results/c4_{slug(model)} --resume  (sbatch hpcc/sbatch/c4_scale.sbatch runs google/gemma-2-9b only)"
    if d is None:
        return cell("not_reached", criterion="FINAL_SPEC 6.2 (T2-T4: certified SAE/probe feature inventory; report certified fraction)",
                    provenance=f"not reached: {why}. Run: {cmd}")
    j = _read_json(os.path.join(d, "C4_summary.json")) or {}
    if not same_model(j.get("model"), model):
        return cell("not_reached", criterion="FINAL_SPEC 6.2",
                    provenance=f"not reached (K-O4-4): {os.path.relpath(d, ROOT)}/C4_summary.json was computed on {j.get('model')!r}, not {model!r}. Run: {cmd}")
    frac = j.get("certified_fraction_alive"); pv = j.get("probe_verdict") or {}
    if frac is None:
        return cell("not_reached", criterion="FINAL_SPEC 6.2", provenance=f"not reached: no certified_fraction_alive in {os.path.relpath(d, ROOT)}/C4_summary.json")
    frac = float(frac)
    status = "failed" if frac == 0 else ("certified" if pv and all(v == "invariant" for v in pv.values()) else "partial")
    return cell(status, value=dict(certified_fraction_alive=frac, alive=j.get("alive"), n_lat=j.get("n_lat"),
                                   layer=j.get("layer"), probe_verdict=pv),
                criterion=("C4's frozen rule: a dictionary feature is certified iff its blind activation-profile match is >= 0.95 under "
                           "EVERY exact renaming sampled. Cell status (SPEC_O4): certified iff fraction > 0 and every probe verdict is "
                           "invariant; partial if some probe verdict is artifact; failed iff fraction == 0. FINAL_SPEC 6.2 sets no numeric "
                           "threshold at T4 — the fraction is reported"),
                provenance=(f"C4 hpcc/sirius_symboli.py, k-sparse dictionary on the MLP-hidden (post-gate) activations at layer "
                            f"{j.get('layer')} — the coordinates the gauge group acts on. A residual-stream Gemma Scope SAE sees no "
                            f"MLP-unit permutation and cannot be gauge-audited, so this inventory is NOT the Gemma Scope inventory. "
                            f"Source {os.path.relpath(d, ROOT)}/C4_summary.json + C4_inventory.csv"
                            + (f". CAVEAT: {note}" if note else "")))


def gamma_cell(results, model, subsample=GAMMA_SUBSAMPLE):
    """Step 7 / tool-side P4: gamma_cov from the K8 Gemma Scope reconstruction residual with the K-block MP nulls."""
    d, note = find_input(results, "k8", model, ["K8_reconstruction.csv"]); why = note
    cmd = (f"python hpcc/biwa_hayahide.py --model {model} --release <SAE release> --out hpcc/results/k8_{slug(model)} --resume"
           if "gemma" not in model.lower() else
           f"sbatch hpcc/sbatch/k8_frontier_residue.sbatch  (python hpcc/biwa_hayahide.py --model {model} --out hpcc/results/k8_{slug(model)} --resume)")
    no_sae = ("no public SAE for this model in this project's stack: Gemma Scope is Gemma-only and sae_lens carries no canonical "
              "Llama-3-8B release that K8 is written against. Without a decomposition there is no residue, so gamma_cov is unknown "
              "and the degree of opening is reported as unknown, never as zero")
    if d is None:
        reason = no_sae if "gemma" not in model.lower() else why
        return cell("not_reached", criterion="FINAL_SPEC 2.13 / 3.7 / 7 R4", provenance=f"not reached: {reason}. Run: {cmd}"), None, None
    rows = [r for r in _read_csv(os.path.join(d, "K8_reconstruction.csv")) if same_model(r.get("model"), model)]
    if not rows:
        return cell("not_reached", criterion="FINAL_SPEC 2.13 / 3.7 / 7 R4",
                    provenance=f"not reached (K-O4-4): {os.path.relpath(d, ROOT)}/K8_reconstruction.csv holds no row for {model!r}. Run: {cmd}"), None, None
    from whiteboxtopo.reconstruct import coverage_gap
    per_layer = {}
    for r in rows:
        L = int(r["layer"]); npz = os.path.join(d, "cache", f"L{L}.npz")
        rec = dict(layer=L, fvu=float(r.get("fvu", "nan")), n_tokens=int(float(r.get("n_tokens", 0))), width=r.get("width"), hook=r.get("hook"))
        if not os.path.exists(npz):
            rec["gamma_cov"] = None; rec["note"] = f"residual cache {os.path.relpath(npz, ROOT)} absent (K8 wrote the row but not the cache)"
        else:
            try:
                Ez = np.load(npz)["E"]
                n = min(subsample, len(Ez))
                sub = np.random.default_rng(GAMMA_SEED).choice(len(Ez), n, replace=False)
                cg = coverage_gap(np.asarray(Ez[np.sort(sub)], dtype=np.float64), q=GAMMA_Q, seed=GAMMA_SEED).to_dict()
                cg.pop("eigenvalues", None); rec.update(cg); rec["n_subsample"] = int(n)
            except Exception as e:
                rec["gamma_cov"] = None; rec["note"] = f"{type(e).__name__}: {e}"
        per_layer[L] = rec
    ok = {L: r for L, r in per_layer.items() if r.get("gamma_cov") is not None}
    if not ok:
        return cell("not_reached", criterion="FINAL_SPEC 2.13", provenance=f"not reached: no usable residual cache in {os.path.relpath(d, ROOT)}/cache; per-layer notes in the evidence",
                    per_layer=per_layer), None, per_layer
    mid = min(ok, key=lambda L: abs(L - (max(ok) + min(ok)) / 2.0))
    head = ok[mid]; g = float(head["gamma_cov"]); exc = float(head.get("gamma_excess", float("nan")))
    status = "certified" if exc > 0 else "failed"
    c = cell(status, value=dict(gamma_cov=g, headline_layer=mid, gamma_excess=exc,
                                gamma_null_gaussian=head.get("gamma_null_gaussian"), gamma_null_shuffle=head.get("gamma_null_shuffle"),
                                fvu=head.get("fvu"), n_outliers=head.get("n_outliers")),
             criterion=(f"gamma_cov = sum of eigenvalues above the Tracy-Widom-corrected MP edge (q = {GAMMA_Q}) over tr Sigma "
                        f"(FINAL_SPEC 2.13), on a fixed {subsample}-row subsample (seed {GAMMA_SEED}) of the SAE residual. "
                        "K-O4-2: claimed only where gamma_excess = gamma_cov - max(gaussian-matched, feature-shuffle nulls) > 0; "
                        "otherwise the residue is MP-indistinguishable and the cell is failed. Headline layer = the K8 layer closest "
                        "to the middle of those run (pre-registered before compute)"),
             provenance=(f"K8 hpcc/biwa_hayahide.py residual E = x - SAE(x) (Gemma Scope, {head.get('width')}, {head.get('hook')}); "
                         f"gamma via whiteboxtopo.reconstruct.coverage_gap; both MP nulls are the K-block nulls verbatim. "
                         f"Source {os.path.relpath(d, ROOT)}/cache/L*.npz + K8_reconstruction.csv"
                         + (f". CAVEAT: {note}" if note else "")),
             per_layer=per_layer)
    return c, (g if status == "certified" else None), per_layer


def l7_cell(results, model):
    d, note = find_input(results, "l7", model, ["L7_timing.json"]); why = note
    if d is None:
        return cell("not_reached", criterion="handbook L7 / FINAL_SPEC 8.1",
                    provenance=f"not reached: {why}. Run: sbatch hpcc/sbatch/l7_end_to_end.sbatch")
    j = _read_json(os.path.join(d, "L7_timing.json")) or {}
    if not same_model(j.get("model"), model):
        return cell("not_reached", criterion="handbook L7",
                    provenance=f"not reached (K-O4-4): L7_timing.json is for {j.get('model')!r}, not {model!r}")
    return cell("certified", value=dict(total_wall_s=j.get("total_wall_s"), does_not_scale=j.get("does_not_scale"),
                                        not_covered=j.get("not_covered"), stages={k: v.get("wall_s") for k, v in (j.get("stages") or {}).items()}),
                criterion="feasibility record, not a P-cell: what the auditor costs end-to-end at 9B and what does not scale",
                provenance=f"L7 hpcc/kawakami_princess.py; source {os.path.relpath(d, ROOT)}/L7_timing.json" + (f". CAVEAT: {note}" if note else ""))


def p0_cell(out_dir, model):
    p = os.path.join(out_dir, f"p0_{slug(model)}.json"); j = _read_json(p)
    cmd = f"sbatch hpcc/sbatch/o41_frontier_p0.sbatch  (python experiments/O_opener/bamboo_memory.py --p0 --model {model} --out hpcc/results/o41 --resume)"
    if j is None:
        return cell("not_reached", criterion="FINAL_SPEC 6.1", provenance=f"not reached: {os.path.relpath(p, ROOT)} absent. Run: {cmd}")
    if not same_model(j.get("model"), model):
        return cell("not_reached", criterion="FINAL_SPEC 6.1", provenance=f"not reached (K-O4-4): {os.path.relpath(p, ROOT)} is for {j.get('model')!r}")
    pr = (j.get("profiles") or {}).get(f"{PREREG_DENSITY}", {}); v = j.get("verdict")
    return cell("certified" if v == "tractable" else "failed",
                value=dict(verdict=v, treewidth_ub=pr.get("treewidth_ub"), Q=pr.get("Q"),
                           Q_null_rewired_mean=pr.get("Q_null_rewired_mean"), Q_excess=pr.get("Q_excess"),
                           sparsity=pr.get("sparsity_node_importance"), density=PREREG_DENSITY,
                           layers=j.get("layers"), n_nodes=j.get("n_nodes"), n_tokens=j.get("n_tokens")),
                criterion=j.get("criterion", ""),
                provenance=(j.get("provenance", "") + f"; source {os.path.relpath(p, ROOT)}"
                            + "; the PRE-REGISTERED graph (read points per layer) sets this verdict. The per-target-node "
                              "EAP repair (#O4.1 (d) / V AMENDMENTS #15 / SPEC_O3 #O3-2) is computed alongside and "
                              "reported in `per_node`, never used to set the verdict"),
                profiles=j.get("profiles"), per_node=j.get("per_node"), gemma_note=j.get("gemma_note"),
                wall_s=j.get("wall_s"))


NOT_REACHED = {
    "P2": ("FINAL_SPEC 6.3 (attention-pattern-conditional path products, reported with the QK caveat)",
           "not reached: needs new code — a frozen-attention re-implementation of the Llama/Gemma forward (RoPE + grouped-query "
           "attention + Gemma-2 logit soft-capping) to form A(x), plus exact-invariance verification of A(x) over a gauge orbit of the "
           "8-9B model. whiteboxtopo.invariants.path_product is the ReLU-MLP instrument and does not apply. Not cheap: one new module "
           "plus O(100-300) renamed forward passes at >= 40 GB, i.e. a job of F12's size. Nothing in hpcc/ computes it"),
    "P3": ("FINAL_SPEC 6.4 (zigzag over checkpoints + Laplacian eigenflow, planted or natural onset)",
           "not reached: needs training checkpoints. Neither meta-llama/Meta-Llama-3-8B nor google/gemma-2-9b publishes intermediate "
           "checkpoints, so this is structural, not a compute shortage. P3 at scale is O3.1's row (Pythia-410M/1B, 154 public revisions, "
           "hpcc/mejiro_ardan.py + the B2/B3 bias-probe procedures)"),
    "P5": ("FINAL_SPEC 6.6 (rho(x) = ||Q_h(x)|| . Lip(x) vs actual failure radius, softmax-margin baseline)",
           "not reached: needs a local second fundamental form on the activation cloud AND the spectral norm of the path product per "
           "input; there is no path product at 8-9B (see P2) and no T4 job computes the curvature field or the perturbation sweep at "
           "six radii. Cost: k-NN PCA over the frontier activation cloud plus 6 x |X| perturbed forward passes"),
    "P6": ("FINAL_SPEC 6.7 (audited memorization_signal per family with baselines, M block)",
           "not reached: whiteboxtopo.signals.memorization_signal needs a family calibration, and the only LM calibration that ships "
           "(signals/calibration/memorization_lm_transformer.json) is arch gpt2, layer L12, n_models 8, whose own meta note reads "
           "'Cross-family transfer FAILS (sign flip); never apply to another family'. M4 (hpcc/admire_vega.py) measures that "
           "cross-family failure and has no 8-9B arm. Applying it here would be a number with no meaning, so it is not computed"),
    "P7": ("FINAL_SPEC 6.8 (zigzag intervals, Laplacian eigenflow, normalised-persistence velocity, legibility drift Q(t))",
           "not reached: needs training checkpoints — same structural reason as P3. O3.1's row"),
}


def assemble(a):
    from whiteboxtopo.open import Scorecard, CellResult
    os.makedirs(a.out, exist_ok=True); sl = slug(a.model); cells = {}
    cells["gauge"], f12_dir, f12_note = gauge_cell(a.results, a.model)
    cells["audit"] = audit_cell(a.results, a.model, f12_dir, f12_note)
    cells["P0"] = p0_cell(a.out, a.model)
    cells["P1"] = p1_cell(a.results, a.model)
    for p in ("P2", "P3", "P5", "P6", "P7"):
        crit, why = NOT_REACHED[p]; cells[p] = cell("not_reached", criterion=crit, provenance=why)
    r4, gamma, per_layer = gamma_cell(a.results, a.model)
    cells["R4"] = r4
    # P4 (FINAL_SPEC 6.5) has two halves; only the tool side is reachable at T4
    model_side = ("the model-side half (diffusion embedding + Nystrom out-of-manifold distance, AUC > 0.9 beating softmax confidence) "
                  "has no T4 job and is not reached")
    if r4["status"] == "certified":
        cells["P4"] = cell("partial", value=dict(tool_side_gamma_cov=r4["value"]["gamma_cov"], headline_layer=r4["value"]["headline_layer"]),
                           criterion="FINAL_SPEC 6.5: tool-side P4 = gamma_cov of a decomposition (K block); " + model_side,
                           provenance="tool side from the R4 cell (K8 Gemma Scope residual); " + model_side)
    else:
        cells["P4"] = cell("not_reached", criterion="FINAL_SPEC 6.5",
                           provenance=f"not reached: tool side unavailable ({r4['provenance']}); " + model_side)
    cells["L7"] = l7_cell(a.results, a.model)

    card = Scorecard(model=a.model, behavior=dict(name="next-token prediction on the fixed OpenWebText-10k probe", kind="general"),
                     config=dict(tier="T4", block="O4.1 Bamboo Memory", spec="experiments/O_opener/SPEC_O4.md",
                                 results_root=os.path.relpath(a.results, ROOT), assembled=time.strftime("%Y-%m-%d %H:%M")))
    for p in CELLS:
        c = cells[p]
        card.cells[p] = CellResult(p, c["status"], value=c["value"], evidence={"criterion": c["criterion"]}, provenance=c["provenance"])
    card.gamma_cov = gamma
    card.reconstruction = {"R4": cells["R4"]["status"], "R1": "not_reached", "R2": "not_reached", "R3": "not_reached",
                           "R5": "not_reached", "R6": "not_reached", "R7": "not_reached", "R8": "not_reached"}
    for k, c in cells.items():
        if c["status"] == "not_reached": card.errors.append(f"{k}: {c['provenance']}")
    card.provenance = {k: c["provenance"][:300] for k, c in cells.items()}

    out = dict(specimen=f"T4 {a.model}", kind="frontier", tier="T4", model=a.model, slug=sl, condition="frontier", seed=None,
               job="O4.1 Bamboo Memory --assemble", spec="experiments/O_opener/SPEC_O4.md",
               cells=_j(cells), scorecard=_j(card.to_dict()), gamma_per_layer=_j(per_layer),
               date=time.strftime("%Y-%m-%d %H:%M"), host=platform.node())
    pj = os.path.join(a.out, f"scorecard_{sl}.json"); json.dump(out, open(pj, "w"), indent=1, default=str)

    deg = card.degree_of_opening
    L = [card.to_markdown(), ""]
    L.append("**Degree of opening (FINAL_SPEC 12.9).** " + (
        f"Opened to degree 1 - gamma_cov = {deg:.4f} (gamma_cov = {card.gamma_cov:.4f} at layer "
        f"{cells['R4']['value']['headline_layer']}). The word \"complete\" is not used: R5-R7 are not run at T4."
        if deg is not None else
        "Degree of opening: **unknown** — no certified decomposition was reached, so the coverage gap is unknown. "
        "An unknown gap is never reported as a zero gap."))
    L += ["", "### Steps 1, 3-4, L7 (not P-cells)", "", "| step | status | value | provenance |", "|---|---|---|---|"]
    for k in ("gauge", "audit", "R4", "L7"):
        c = cells[k]; v = json.dumps(c["value"], default=str)
        L.append(f"| {k} | {c['status']} | {v[:110].replace('|', '/')} | {c['provenance'][:150].replace('|', '/')} |")
    if per_layer:
        L += ["", "### gamma_cov per K8 layer", "", "| layer | fvu | gamma_cov | gaussian null | shuffle null | gamma_excess |", "|---|---|---|---|---|---|"]
        for Lx in sorted(per_layer):
            r = per_layer[Lx]
            f = lambda k: ("—" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k))))
            L.append(f"| {Lx} | {f('fvu')} | {f('gamma_cov')} | {f('gamma_null_gaussian')} | {f('gamma_null_shuffle')} | {f('gamma_excess')} |"
                     + (f"  <!-- {r['note']} -->" if r.get("note") else ""))
    L += ["", "### Cells not reached, and why", ""]
    for p in CELLS:
        if cells[p]["status"] == "not_reached": L.append(f"* **{p}** — {cells[p]['provenance']}")
    L += ["", f"Pre-registration: `experiments/O_opener/SPEC_O4.md`. Evidence: `{os.path.relpath(pj, ROOT)}`.", ""]
    pm = os.path.join(a.out, f"scorecard_{sl}.md"); open(pm, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[assemble] wrote {pj}\n[assemble] wrote {pm}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="O4.1 Bamboo Memory — T4 frontier open() (SPEC_O4.md)")
    ap.add_argument("--p0", action="store_true", help="compute the P0 attribution-graph cell on the frontier model (GPU)")
    ap.add_argument("--no-per-node", action="store_true", help="skip the per-target-node EAP graph (#O4.1 (d) repair); "
                                                               "the pre-registered graph and the P0 verdict are unaffected")
    ap.add_argument("--n-gauge", type=int, default=3, help="K-O-2 gauge samples for the P0 graphs (0 = skip); needs a "
                                                           "whiteboxtopo.symmetry backend for the model")
    ap.add_argument("--gauge-seq", type=int, default=4, help="sequences of the probe used for the K-O-2 orbit (0 = all). "
                                                             "A cost bound: the orbit measures gauge sensitivity, not the "
                                                             "reported numbers")
    ap.add_argument("--assemble", action="store_true", help="read the frontier jobs' outputs and write the scorecard (CPU)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="explicit layer list whose heads + MLP become graph nodes (default: mid-2 .. mid+2). A single layer gives "
                         "only within-layer head->MLP edges")
    ap.add_argument("--n-tokens", type=int, default=2000, help="probe size (pre-registered 2000 at T4, as V7)")
    ap.add_argument("--seq-len", type=int, default=126)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-null", type=int, default=20, help="degree-preserving rewirings for the Q null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="--assemble: root holding f12_*/c4*/k8*/l7*/o41")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if not (a.p0 or a.assemble): ap.error("pass --p0 and/or --assemble")
    if a.p0: run_p0(a)
    if a.assemble: assemble(a)


if __name__ == "__main__":
    main()
