"""Llama / Gemma-2 family backend (HF ``LlamaForCausalLM``, ``Gemma2ForCausalLM``) for the symmetry group (F12).

Exact renamings of a decoder block with RMSNorm, RoPE grouped-query attention and a gated MLP:
* ``("mlp", l)``     MLP hidden-unit permutation (rows of gate_proj/up_proj, columns of down_proj)
* ``("kvhead", l)``  permutation of KV heads *together with* their query groups
* ``("qhead", l)``   permutation of query heads inside each KV group (q_proj rows of the group; o_proj columns)
* ``("vo", l)``      per-KV-head invertible change of basis: v_proj block -> R v (rows), o_proj columns of the group -> R^{-1}
* ``("qk", l)``      per-head RoPE-commuting change of basis: on every rotary 2-plane (i, i + d_head/2) a complex
                     scalar s·e^{iφ} on Q and (1/s)·e^{iφ} on K — the only linear maps commuting with all RoPE rotations
                     (plane permutations are NOT symmetries: RoPE frequencies differ per plane)
* ``("ln1", l)``     input RMSNorm weight rescale (γ_i -> a_i γ_i; column i of q/k/v_proj / a_i)
* ``("ln2", l)``     post-attention RMSNorm rescale (γ_i -> a_i γ_i; column i of gate/up_proj / a_i)
Gemma-2 stores RMSNorm as (1 + w): the rescale acts on (1 + w) and the compensation is identical.
NOT exact and NOT sampled: MLP unit scalings (SiLU/GELU gates are not homogeneous); final norm rescale (tied
embeddings in Gemma; Llama-3.2-1B ties too). Attention logit soft-capping (Gemma-2) is a function of the
q·k product and is unaffected by the maps above.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from .group import GaugeElement
from .transformer import _rand_rot


def handles(model) -> bool:
    return type(model).__name__ in ("LlamaForCausalLM", "Gemma2ForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM")


def _cfg(model):
    c = model.config; H = c.num_attention_heads; KV = getattr(c, "num_key_value_heads", H) or H
    dh = getattr(c, "head_dim", None) or c.hidden_size // H
    return dict(L=c.num_hidden_layers, H=H, KV=KV, dh=dh, d=c.hidden_size, m=c.intermediate_size, group=H // KV,
                gemma=type(model).__name__.startswith("Gemma"))


def describe(model):
    c = _cfg(model)
    return {"type": "llama_family", **c, "known_dim": int(c["L"] * (c["KV"] * c["dh"] ** 2 + c["H"] * c["dh"] + 2 * c["d"])),
            "exact_scalings": ["ln1", "ln2", "qk (per-plane complex scalar)"], "approximate_or_absent": ["mlp (gated, not homogeneous)", "final norm (tied embeddings)"],
            "keys": [(k, l) for l in range(c["L"]) for k in ("mlp", "kvhead", "qhead", "vo", "qk", "ln1", "ln2")],
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def sample(model, rng, kind="combined", log_range=(0.1, 10.0), orthogonal=True, layers=None) -> GaugeElement:
    c = _cfg(model); g = GaugeElement(kind=kind); lo, hi = np.log(log_range[0]), np.log(log_range[1])
    for l in (range(c["L"]) if layers is None else layers):
        if kind in ("perm", "combined"):
            g.perms[("mlp", l)] = rng.permutation(c["m"]); g.perms[("kvhead", l)] = rng.permutation(c["KV"])
            g.perms[("qhead", l)] = np.stack([rng.permutation(c["group"]) for _ in range(c["KV"])])
        if kind in ("scale", "combined"):
            g.scales[("ln1", l)] = np.exp(rng.uniform(lo, hi, size=c["d"])); g.scales[("ln2", l)] = np.exp(rng.uniform(lo, hi, size=c["d"]))
            g.scales[("qk", l)] = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(c["H"], c["dh"] // 2)))   # per-plane scale s
        if kind in ("rot", "combined"):
            g.rots[("vo", l)] = [_rand_rot(rng, c["dh"], orthogonal) for _ in range(c["KV"])]
            g.rots[("qk", l)] = rng.uniform(0, 2 * np.pi, size=(c["H"], c["dh"] // 2))                         # per-plane phase φ
    return g


def uniform_scaling(model, alpha, layer=None):
    raise NotImplementedError("gated MLPs have no exact homogeneous unit scaling")


def _plane_matrix(dh, s, phi, dtype, device):
    """Block matrix C acting on a head vector in HF rotate_half layout: plane j = (j, j + dh/2)."""
    half = dh // 2; C = torch.zeros(dh, dh, dtype=torch.float64)
    for j in range(half):
        cj, sj = np.cos(phi[j]) * s[j], np.sin(phi[j]) * s[j]
        C[j, j] = cj; C[j, j + half] = -sj; C[j + half, j] = sj; C[j + half, j + half] = cj
    return C.to(dtype=dtype, device=device)


@torch.no_grad()
def apply(model, g: GaugeElement):
    new = copy.deepcopy(model); c = _cfg(new); H, KV, dh, d, m, G = c["H"], c["KV"], c["dh"], c["d"], c["m"], c["group"]
    for l, blk in enumerate(new.model.layers):
        at, mlp = blk.self_attn, blk.mlp
        Wq, Wk, Wv, Wo = at.q_proj.weight, at.k_proj.weight, at.v_proj.weight, at.o_proj.weight   # (H*dh, d), (KV*dh, d), (KV*dh, d), (d, H*dh)
        bq = getattr(at.q_proj, "bias", None); bk = getattr(at.k_proj, "bias", None); bv = getattr(at.v_proj, "bias", None)
        # --- qk: per-plane complex scalars (commute with RoPE): Q_h -> C_h Q_h (rows of q_proj block), K_kv -> C^{-T}... K is shared
        # across the group, so the K transform must be the same for every query head of the group: use head 0's (s, φ) for the
        # whole group and set Q heads of the group to the matching C.
        if ("qk", l) in g.rots or ("qk", l) in g.scales:
            phi = g.rots.get(("qk", l), np.zeros((H, dh // 2))); s = g.scales.get(("qk", l), np.ones((H, dh // 2)))
            for kv in range(KV):
                h0 = kv * G; C = _plane_matrix(dh, s[h0], phi[h0], Wq.dtype, Wq.device); Cinv_T = torch.linalg.inv(C.double()).T.to(Wq.dtype)
                for h in range(h0, h0 + G):
                    sl = slice(h * dh, (h + 1) * dh); Wq[sl] = C @ Wq[sl]
                    if bq is not None: bq[sl] = C @ bq[sl]
                sk = slice(kv * dh, (kv + 1) * dh); Wk[sk] = Cinv_T @ Wk[sk]
                if bk is not None: bk[sk] = Cinv_T @ bk[sk]
        # --- vo: per-KV-head invertible R: v -> R v (rows of v_proj block), o_proj columns of every query head in the group -> R^{-1}
        if ("vo", l) in g.rots:
            for kv, Rm in enumerate(g.rots[("vo", l)]):
                Rt = torch.as_tensor(Rm, dtype=torch.float64, device=Wv.device); Rinv = torch.linalg.inv(Rt)
                sk = slice(kv * dh, (kv + 1) * dh); Wv[sk] = (Rt @ Wv[sk].double()).to(Wv.dtype)
                if bv is not None: bv[sk] = (Rt @ bv[sk].double()).to(bv.dtype)
                for h in range(kv * G, (kv + 1) * G):
                    so = slice(h * dh, (h + 1) * dh); Wo[:, so] = (Wo[:, so].double() @ Rinv).to(Wo.dtype)
        # --- head permutations
        if ("qhead", l) in g.perms or ("kvhead", l) in g.perms:
            pq = g.perms.get(("qhead", l), np.tile(np.arange(G), (KV, 1))); pkv = g.perms.get(("kvhead", l), np.arange(KV))
            # new query head order: for new kv slot j (old kv pkv[j]), query heads old = pkv[j]*G + pq[pkv[j]][i]
            order = np.concatenate([[int(pkv[j]) * G + int(pq[int(pkv[j])][i]) for i in range(G)] for j in range(KV)])
            qrows = np.concatenate([np.arange(h * dh, (h + 1) * dh) for h in order]); kvrows = np.concatenate([np.arange(k * dh, (k + 1) * dh) for k in pkv])
            qi = torch.as_tensor(qrows, device=Wq.device); ki = torch.as_tensor(kvrows, device=Wq.device)
            Wq.copy_(Wq[qi]); Wk.copy_(Wk[ki]); Wv.copy_(Wv[ki]); Wo.copy_(Wo[:, qi])
            if bq is not None: bq.copy_(bq[qi])
            if bk is not None: bk.copy_(bk[ki])
            if bv is not None: bv.copy_(bv[ki])
        # --- MLP permutation
        if ("mlp", l) in g.perms:
            p = torch.as_tensor(np.asarray(g.perms[("mlp", l)]), device=Wq.device)
            mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[p]); mlp.up_proj.weight.copy_(mlp.up_proj.weight[p]); mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, p])
        # --- RMSNorm rescales
        if ("ln1", l) in g.scales:
            a = torch.as_tensor(g.scales[("ln1", l)], dtype=Wq.dtype, device=Wq.device); w = blk.input_layernorm.weight
            if c["gemma"]: w.copy_((1 + w) * a - 1)
            else: w.mul_(a)
            for W in (Wq, Wk, Wv): W.div_(a[None, :])
        if ("ln2", l) in g.scales:
            a = torch.as_tensor(g.scales[("ln2", l)], dtype=Wq.dtype, device=Wq.device); w = blk.post_attention_layernorm.weight
            if c["gemma"]: w.copy_((1 + w) * a - 1)
            else: w.mul_(a)
            mlp.gate_proj.weight.div_(a[None, :]); mlp.up_proj.weight.div_(a[None, :])
    return new


@torch.no_grad()
def forward(model, tokens):
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=next(model.parameters()).device)
    if tokens.ndim == 1: tokens = tokens[None]
    return model(tokens).logits.float().cpu().numpy()


def param_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().double().numpy()
