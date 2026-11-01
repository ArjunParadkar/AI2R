"""GPT-NeoX / Pythia backend for the symmetry group (FINAL_SPEC §2.3, §3 step 1).

Exact renamings of ``GPTNeoXForCausalLM`` / ``GPTNeoXModel``. The block is

    h  = hidden_states                                   (residual stream, width d)
    a  = dense( attn( input_layernorm(h) ) )
    m  = dense_4h_to_h( GELU( dense_h_to_4h( post_attention_layernorm(h) ) ) )
    h' = h + a + m          (``use_parallel_residual=True``, Pythia)

so *attention and MLP read the same residual stream but through two separate
LayerNorms*: ``input_layernorm`` has exactly one consumer (``query_key_value``)
and ``post_attention_layernorm`` has exactly one consumer (``dense_h_to_4h``).
(With ``use_parallel_residual=False`` the second LayerNorm reads ``h + a``
instead — still one consumer. Either way the compensation below is the same.)

The residual stream itself is *not* re-based: every renaming here is internal to
one layer (head space, MLP hidden space, or a LayerNorm output that is
immediately consumed by one linear map), so the residual stream, the embeddings
and the unembedding are untouched and the function is unchanged exactly.

Keys (``GaugeElement`` dict entries):

* ``("head", l)``    perms  — permutation of the H attention heads. ``query_key_value``
  is a single ``Linear(d, 3d)`` whose *rows* are laid out head-major as
  ``[q_h | k_h | v_h]`` for h = 0..H-1 (block ``h`` = rows ``h*3*dh : (h+1)*3*dh``);
  ``dense`` consumes the heads head-major in its input (= *columns* of
  ``dense.weight``, which is ``(d, H*dh)``). Permuting both is exact.
* ``("vo", l)``      rots   — per-head invertible ``B ∈ GL(d_head)`` on the value/output
  pair: ``W_V ↦ B W_V`` (rows of the v block, and its bias), ``W_O ↦ W_O B^{-1}``
  (the head's columns of ``dense``). Exact: attention mixes values linearly, so
  ``Σ_s α_s (B v_s)`` then ``B^{-1}`` gives back ``Σ_s α_s v_s``.
* ``("qk", l)``      rots (phase φ) + scales (s) — the RoPE-commuting part of the Q/K
  change of basis, on the **first ``rotary_ndims`` dims of each head only**
  (``rotary_pct = 0.25`` in Pythia ⇒ 16 of 64). On every rotary 2-plane
  ``(j, j + rotary_ndims/2)`` (HF ``rotate_half`` layout) the map is the complex
  scalar ``s_j e^{iφ_j}`` on Q and ``(1/s_j) e^{iφ_j}`` on K. These are *exactly*
  the linear maps of the rotary subspace that commute with every RoPE rotation,
  because the per-plane frequencies are distinct — plane permutations are NOT
  symmetries.
* ``("qkpass", l)``  rots   — per-head **unrestricted** ``A ∈ GL(d_head - rotary_ndims)``
  on the non-rotary tail: ``W_Q ↦ A W_Q``, ``W_K ↦ A^{-T} W_K``. Exact, because
  RoPE never touches those coordinates and ``(Aq)·(A^{-T}k) = q·k``.
  Absent when ``rotary_pct == 1``.
  The full per-head Q/K map is ``M = blockdiag(rotary planes, A)``; block-diagonal
  is not a convenience but the whole group: a block mixing the rotary and
  non-rotary parts leaves an uncancelled position-dependent cross term.
* ``("mlp", l)``     perms  — MLP hidden-unit permutation: rows of ``dense_h_to_4h``
  and its bias, columns of ``dense_4h_to_h``. Exact.
* ``("ln1", l)``     scales — ``input_layernorm`` gain/bias rescale ``(γ_i, β_i) ↦ a_i (γ_i, β_i)``
  with column i of ``query_key_value`` divided by ``a_i``. Exact.
* ``("ln2", l)``     scales — the same for ``post_attention_layernorm`` / ``dense_h_to_4h``.
* ``"lnf"``          scales — ``final_layer_norm`` rescale with the columns of the unembedding
  (``lm_head`` in current transformers, ``embed_out`` in older ones) divided by ``a``.
  Exact **only because Pythia has untied embeddings** (``tie_word_embeddings=False``);
  sampled only when that matrix exists and is not tied to ``embed_in`` — this is the
  one symmetry GPT-2 does *not* have. Absent for a bare ``GPTNeoXModel`` (whose output
  *is* the final hidden state).

NOT exact and therefore NOT sampled:

* MLP hidden-unit **scalings** — the activation is GELU, which is not positively
  homogeneous, so ``(w_in, b) ↦ a (w_in, b)``, ``w_out ↦ w_out / a`` changes the
  function for ``a ≠ 1``. ``uniform_scaling`` therefore raises.
* Any re-basing of the residual stream (it is read by every layer and by the
  untied unembedding, and LayerNorm's mean-subtraction/normalisation is not
  equivariant under a general linear map).
* Rotary-plane permutations and any Q/K map mixing rotary with non-rotary dims
  (see above).

Everything listed as a key is exact in exact arithmetic; in fp32 the measured
floor on pythia-70m is ~1e-4 relative (``tests/test_neox.py``). The Linear
convention here is ``weight (out, in)`` — note this is the *transpose* of the
GPT-2 ``Conv1D`` convention used in ``gpt2.py``.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from .group import GaugeElement
from .transformer import _rand_rot


def handles(model) -> bool:
    return type(model).__name__ in ("GPTNeoXForCausalLM", "GPTNeoXModel")


def _core(model):
    """The ``GPTNeoXModel`` inside (or the model itself)."""
    return getattr(model, "gpt_neox", model)


def _blocks(model):
    return _core(model).layers


def _lm_head(model):
    """The unembedding (``embed_out`` in old transformers, ``lm_head`` in new) if it exists
    and is *not* tied to ``embed_in``, else None."""
    head = getattr(model, "embed_out", None) or getattr(model, "lm_head", None)
    if head is None and hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        return None
    if getattr(model.config, "tie_word_embeddings", False):
        return None
    if head.weight.data_ptr() == _core(model).embed_in.weight.data_ptr():
        return None
    return head


def _cfg(model):
    c = model.config
    H = c.num_attention_heads
    d = c.hidden_size
    dh = getattr(c, "head_dim", None) or d // H
    pct = 1.0
    rp = getattr(c, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("partial_rotary_factor") is not None:
        pct = float(rp["partial_rotary_factor"])
    elif getattr(c, "partial_rotary_factor", None) is not None:
        pct = float(c.partial_rotary_factor)
    elif getattr(c, "rotary_pct", None) is not None:
        pct = float(c.rotary_pct)
    r = int(dh * pct)
    return dict(L=c.num_hidden_layers, H=H, dh=dh, d=d, m=c.intermediate_size, r=r, pct=pct,
                n_pass=dh - r, parallel=bool(getattr(c, "use_parallel_residual", True)),
                untied=_lm_head(model) is not None)


def describe(model):
    c = _cfg(model)
    keys = [(k, l) for l in range(c["L"]) for k in
            (("head", "vo", "qk") + (("qkpass",) if c["n_pass"] > 0 else ()) + ("mlp", "ln1", "ln2"))]
    if c["untied"]:
        keys.append("lnf")
    # continuous dimension of the known group: per layer GL(dh) on V/O, (scale, phase) per rotary
    # plane, GL(n_pass) on the Q/K tail, and two LayerNorm gain vectors; plus the final LN if untied.
    per_layer = c["H"] * c["dh"] ** 2 + c["H"] * (c["r"] // 2) * 2 + c["H"] * c["n_pass"] ** 2 + 2 * c["d"]
    return {"type": "gpt_neox", "n_layers": c["L"], "n_heads": c["H"], "d_head": c["dh"], "d_model": c["d"],
            "d_mlp": c["m"], "rotary_ndims": c["r"], "rotary_pct": c["pct"], "n_pass": c["n_pass"],
            "use_parallel_residual": c["parallel"], "untied_unembedding": c["untied"], "keys": keys,
            "known_dim": int(c["L"] * per_layer + (c["d"] if c["untied"] else 0)),
            "exact_scalings": ["ln1", "ln2", "qk (per-rotary-plane complex scalar)"]
                              + (["lnf (untied embed_out)"] if c["untied"] else []),
            "approximate_or_absent": ["mlp unit scalings (GELU is not positively homogeneous)",
                                      "residual-stream re-basing (LayerNorm + untied unembedding)",
                                      "rotary-plane permutations / rotary-to-tail mixing (RoPE frequencies differ)"]
                                     + ([] if c["untied"] else ["lnf (tied embeddings)"]),
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def sample(model, rng, kind="combined", log_range=(0.1, 10.0), orthogonal=True, layers=None) -> GaugeElement:
    c = _cfg(model)
    H, dh, d, m, r, npass = c["H"], c["dh"], c["d"], c["m"], c["r"], c["n_pass"]
    g = GaugeElement(kind=kind)
    lo, hi = np.log(log_range[0]), np.log(log_range[1])
    for l in (range(c["L"]) if layers is None else layers):
        if kind in ("perm", "combined"):
            g.perms[("head", l)] = rng.permutation(H)
            g.perms[("mlp", l)] = rng.permutation(m)
        if kind in ("scale", "combined"):
            g.scales[("ln1", l)] = np.exp(rng.uniform(lo, hi, size=d))
            g.scales[("ln2", l)] = np.exp(rng.uniform(lo, hi, size=d))
            if r >= 2:
                g.scales[("qk", l)] = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(H, r // 2)))
        if kind in ("rot", "combined"):
            g.rots[("vo", l)] = [_rand_rot(rng, dh, orthogonal) for _ in range(H)]
            if r >= 2:
                g.rots[("qk", l)] = rng.uniform(0, 2 * np.pi, size=(H, r // 2))
            if npass > 0:
                g.rots[("qkpass", l)] = [_rand_rot(rng, npass, orthogonal) for _ in range(H)]
    if kind in ("scale", "combined") and c["untied"] and layers is None:
        g.scales["lnf"] = np.exp(rng.uniform(lo, hi, size=d))
    return g


def uniform_scaling(model, alpha, layer=None) -> GaugeElement:
    raise NotImplementedError("GPT-NeoX has no exact homogeneous per-unit scaling (GELU MLP); "
                              "use the ln1/ln2 LayerNorm rescalings")


def _qk_matrix(dh, r, s, phi, A):
    """The per-head Q/K map M = blockdiag(rotary planes (s, φ), A on the non-rotary tail).

    Rotary layout is HF ``rotate_half``: plane j is the coordinate pair (j, j + r/2)
    *within the first r dims of the head*."""
    M = torch.zeros(dh, dh, dtype=torch.float64)
    half = r // 2
    for j in range(half):
        cj, sj = float(np.cos(phi[j]) * s[j]), float(np.sin(phi[j]) * s[j])
        M[j, j] = cj; M[j, j + half] = -sj
        M[j + half, j] = sj; M[j + half, j + half] = cj
    if dh > r:
        M[r:, r:] = torch.as_tensor(np.asarray(A, dtype=np.float64))
    return M


@torch.no_grad()
def apply(model, g: GaugeElement):
    new = copy.deepcopy(model)
    c = _cfg(new)
    H, dh, d, m, r, npass = c["H"], c["dh"], c["d"], c["m"], c["r"], c["n_pass"]
    eye_pass = np.eye(npass) if npass > 0 else None
    for l, blk in enumerate(_blocks(new)):
        at, mlp = blk.attention, blk.mlp
        Wqkv = at.query_key_value.weight                     # (3d, d), rows head-major [q|k|v]
        bqkv = getattr(at.query_key_value, "bias", None)     # (3d,)
        Wo = at.dense.weight                                 # (d, H*dh), columns head-major
        dev = Wqkv.device

        # --- (ii)+(iii) per-head basis changes: M on Q, M^{-T} on K, B on V, B^{-1} on O ---
        touch = any(k in dd for k, dd in ((("qk", l), g.rots), (("qk", l), g.scales),
                                          (("qkpass", l), g.rots), (("vo", l), g.rots)))
        if touch:
            phi = g.rots.get(("qk", l), np.zeros((H, max(r // 2, 1))))
            sc = g.scales.get(("qk", l), np.ones((H, max(r // 2, 1))))
            Ap = g.rots.get(("qkpass", l), [eye_pass] * H if npass > 0 else [None] * H)
            Bs = g.rots.get(("vo", l), [np.eye(dh)] * H)
            for h in range(H):
                M = _qk_matrix(dh, r, sc[h], phi[h], Ap[h])
                MinvT = torch.linalg.inv(M).T
                B = torch.as_tensor(np.asarray(Bs[h], dtype=np.float64))
                Binv = torch.linalg.inv(B)
                base = h * 3 * dh
                q = slice(base, base + dh); k = slice(base + dh, base + 2 * dh); v = slice(base + 2 * dh, base + 3 * dh)
                M, MinvT, B = M.to(dev), MinvT.to(dev), B.to(dev)
                Wqkv[q] = (M @ Wqkv[q].double()).to(Wqkv.dtype)
                Wqkv[k] = (MinvT @ Wqkv[k].double()).to(Wqkv.dtype)
                Wqkv[v] = (B @ Wqkv[v].double()).to(Wqkv.dtype)
                if bqkv is not None:
                    bqkv[q] = (M @ bqkv[q].double()).to(bqkv.dtype)
                    bqkv[k] = (MinvT @ bqkv[k].double()).to(bqkv.dtype)
                    bqkv[v] = (B @ bqkv[v].double()).to(bqkv.dtype)
                so = slice(h * dh, (h + 1) * dh)
                Wo[:, so] = (Wo[:, so].double() @ Binv.to(dev)).to(Wo.dtype)

        # --- (i) head permutation: rows of query_key_value, columns of dense ---
        if ("head", l) in g.perms:
            p = np.asarray(g.perms[("head", l)])
            rows = torch.as_tensor(np.concatenate([np.arange(j * 3 * dh, (j + 1) * 3 * dh) for j in p]), device=dev)
            cols = torch.as_tensor(np.concatenate([np.arange(j * dh, (j + 1) * dh) for j in p]), device=dev)
            Wqkv.copy_(Wqkv[rows]); Wo.copy_(Wo[:, cols])
            if bqkv is not None:
                bqkv.copy_(bqkv[rows])
            # at.dense.bias lives in residual-stream coordinates: unchanged.

        # --- (iv) MLP hidden-unit permutation (no scalings: GELU) ---
        if ("mlp", l) in g.perms:
            p = torch.as_tensor(np.asarray(g.perms[("mlp", l)]), device=dev)
            mlp.dense_h_to_4h.weight.copy_(mlp.dense_h_to_4h.weight[p])
            if getattr(mlp.dense_h_to_4h, "bias", None) is not None:
                mlp.dense_h_to_4h.bias.copy_(mlp.dense_h_to_4h.bias[p])
            mlp.dense_4h_to_h.weight.copy_(mlp.dense_4h_to_h.weight[:, p])
            # dense_4h_to_h.bias is in residual-stream coordinates: unchanged.

        # --- (v) LayerNorm gain/bias rescalings, compensated in the single consumer ---
        if ("ln1", l) in g.scales:
            a = torch.as_tensor(np.asarray(g.scales[("ln1", l)]), dtype=Wqkv.dtype, device=dev)
            blk.input_layernorm.weight.mul_(a); blk.input_layernorm.bias.mul_(a)
            Wqkv.div_(a[None, :])
        if ("ln2", l) in g.scales:
            a = torch.as_tensor(np.asarray(g.scales[("ln2", l)]), dtype=Wqkv.dtype, device=dev)
            blk.post_attention_layernorm.weight.mul_(a); blk.post_attention_layernorm.bias.mul_(a)
            mlp.dense_h_to_4h.weight.div_(a[None, :])

    head = _lm_head(new)
    if "lnf" in g.scales and head is not None:
        ln = _core(new).final_layer_norm
        a = torch.as_tensor(np.asarray(g.scales["lnf"]), dtype=ln.weight.dtype, device=ln.weight.device)
        ln.weight.mul_(a); ln.bias.mul_(a)
        head.weight.div_(a[None, :])
    return new


@torch.no_grad()
def forward(model, tokens):
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=next(model.parameters()).device)
    if tokens.ndim == 1:
        tokens = tokens[None]
    out = model(tokens)
    y = getattr(out, "logits", None)
    if y is None:
        y = out.last_hidden_state
    return y.float().cpu().numpy()


def param_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().double().numpy()
