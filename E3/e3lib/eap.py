"""Edge attribution patching (EAP) per prompt on the GPT-2-small heads+MLPs graph.

Graph as in Syed, Rager & Conmy 2023 / Hanna et al. 2024: upstream nodes = input,
144 heads, 12 MLPs (157); downstream inputs = head q/k/v (432), MLP-in (12),
logits (1) (445); causally allowed pairs → 32 491 edges.

Per-prompt score s_p(u→v) = Σ_pos (a_u^corr − a_u^clean) · ∂metric_p/∂in_v^clean
(gradient on the clean run; Nanda 2023 attribution patching, edge form).
"""
from __future__ import annotations

import numpy as np
import torch

N_LAYERS, N_HEADS, D_MODEL = 12, 12, 768


class EdgeGraph:
    def __init__(self, n_layers=N_LAYERS, n_heads=N_HEADS):
        self.n_layers, self.n_heads = n_layers, n_heads
        per = n_heads + 1
        # upstream
        self.up_names = ["input"]
        for L in range(n_layers):
            self.up_names += [f"a{L}.h{h}" for h in range(n_heads)] + [f"m{L}"]
        # downstream
        self.down_names = []
        for L in range(n_layers):
            for h in range(n_heads):
                self.down_names += [f"a{L}.h{h}<{c}" for c in "qkv"]
        self.down_names += [f"m{L}" for L in range(n_layers)] + ["logits"]
        self.n_up, self.n_down = len(self.up_names), len(self.down_names)
        # allowed-upstream count per downstream node
        n_up_allowed = []
        for L in range(n_layers):
            n_up_allowed += [1 + per * L] * (3 * n_heads)
        n_up_allowed += [1 + per * L + n_heads for L in range(n_layers)]
        n_up_allowed += [self.n_up]
        self.n_up_allowed = np.array(n_up_allowed)
        us, vs = [], []
        for v, nu in enumerate(self.n_up_allowed):
            us += list(range(nu)); vs += [v] * nu
        self.edge_u = np.array(us); self.edge_v = np.array(vs)
        self.n_edges = len(us)
        self.edge_names = [f"{self.up_names[u]}->{self.down_names[v]}" for u, v in zip(us, vs)]

    def edge_index(self, u_name, v_name):
        return self.edge_names.index(f"{u_name}->{v_name}")


def _up_hooks(n_layers):
    hooks = ["blocks.0.hook_resid_pre"]
    for L in range(n_layers):
        hooks += [f"blocks.{L}.attn.hook_result", f"blocks.{L}.hook_mlp_out"]
    return hooks


def _down_hooks(n_layers):
    hooks = []
    for L in range(n_layers):
        hooks += [f"blocks.{L}.hook_q_input", f"blocks.{L}.hook_k_input", f"blocks.{L}.hook_v_input"]
    hooks += [f"blocks.{L}.hook_mlp_in" for L in range(n_layers)]
    hooks += [f"blocks.{n_layers-1}.hook_resid_post"]
    return hooks


def _stack_up(cache, n_layers):
    """-> [b, pos, n_up, d]"""
    parts = [cache["blocks.0.hook_resid_pre"][:, :, None, :]]
    for L in range(n_layers):
        parts.append(cache[f"blocks.{L}.attn.hook_result"])            # [b,pos,heads,d]
        parts.append(cache[f"blocks.{L}.hook_mlp_out"][:, :, None, :])
    return torch.cat(parts, dim=2)


def _stack_down(grads, n_layers):
    """-> [b, pos, n_down, d] in EdgeGraph.down order"""
    parts = []
    for L in range(n_layers):
        q, k, v = (grads[f"blocks.{L}.hook_{c}_input"] for c in "qkv")   # [b,pos,heads,d]
        parts.append(torch.stack([q, k, v], dim=3).flatten(2, 3))        # heads x qkv interleaved
    for L in range(n_layers):
        parts.append(grads[f"blocks.{L}.hook_mlp_in"][:, :, None, :])
    parts.append(grads[f"blocks.{n_layers-1}.hook_resid_post"][:, :, None, :])
    return torch.cat(parts, dim=2)


def prepare_model(model):
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _corrupt_up(model, toks, n_layers):
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in set(_up_hooks(n_layers)))
    return _stack_up(cache, n_layers)


def eap_batch(model, toks_clean, toks_corr, metric_fn, graph: EdgeGraph, emb_layers=(6, 11)):
    """Returns per-prompt edge scores [b, n_edges] (float32, cpu), embeddings dict
    {layer: [b, d]} (mean over positions of resid_post, clean run), metric clean/corrupt [b]."""
    nl = graph.n_layers
    U_corr = _corrupt_up(model, toks_corr, nl)
    with torch.no_grad():
        logits_c = model(toks_corr)
        metric_corr = metric_fn(logits_c).detach().cpu()
    store, keep = {}, {}
    up_set, down_set = set(_up_hooks(nl)), set(_down_hooks(nl))
    emb_set = {f"blocks.{L}.hook_resid_post" for L in emb_layers}

    def fwd(t, hook):
        if hook.name == "blocks.0.hook_resid_pre":
            t = t.detach().requires_grad_(True)   # params are frozen: make the graph start here
        if hook.name in down_set:
            t.retain_grad(); keep[hook.name] = t
        if hook.name in up_set:
            store[hook.name] = t.detach()
        if hook.name in emb_set:
            store["emb", hook.name] = t.detach().mean(dim=1)
        return t

    names = up_set | down_set | emb_set
    with torch.enable_grad():
        logits = model.run_with_hooks(toks_clean, fwd_hooks=[(lambda n: n in names, fwd)])
        metric = metric_fn(logits)
        metric.sum().backward()
    U_clean = _stack_up(store, nl)
    grads = {n: keep[n].grad for n in down_set}
    D = _stack_down(grads, nl)                                    # [b,pos,n_down,d]
    dU = (U_corr - U_clean)                                       # [b,pos,n_up,d]
    b = D.shape[0]
    scores = torch.empty(b, graph.n_edges, device=D.device)
    # chunk over downstream nodes to bound memory
    chunk = 64
    for v0 in range(0, graph.n_down, chunk):
        v1 = min(v0 + chunk, graph.n_down)
        full = torch.einsum("bpud,bpvd->buv", dU, D[:, :, v0:v1])   # [b, n_up, chunk]
        sel = (graph.edge_v >= v0) & (graph.edge_v < v1)
        scores[:, sel] = full[:, graph.edge_u[sel], graph.edge_v[sel] - v0]
    embs = {L: store["emb", f"blocks.{L}.hook_resid_post"].cpu() for L in emb_layers}
    del keep, store, grads, D, dU, U_corr, U_clean
    return scores.cpu(), embs, metric.detach().cpu(), metric_corr


def ioi_metric(batch):
    ans = torch.tensor([p["answer_tok"] for p in batch]); wrong = torch.tensor([p["wrong_tok"] for p in batch])

    def f(logits):
        last = logits[:, -1]
        return last[torch.arange(len(batch)), ans.to(last.device)] - last[torch.arange(len(batch)), wrong.to(last.device)]
    return f


def induction_metric(batch):
    ans = torch.tensor([p["answer_tok"] for p in batch])

    def f(logits):
        last = logits[:, -1]
        return last[torch.arange(len(batch)), ans.to(last.device)]
    return f


def tokenize(model, prompts):
    """Attach `tokens`/`tokens_corrupt` (int lists) to string prompts; check equal lengths."""
    for p in prompts:
        if "tokens" not in p:
            p["tokens"] = model.to_tokens(p["clean"])[0].tolist()
            p["tokens_corrupt"] = model.to_tokens(p["corrupt"])[0].tolist()
        assert len(p["tokens"]) == len(p["tokens_corrupt"]), p
    return prompts


def eap_all(model, prompts, metric_factory, graph: EdgeGraph, batch_size=8, emb_layers=(6, 11),
            device="cuda", log=print):
    """Per-prompt EAP scores for every prompt (batched by token length)."""
    prompts = tokenize(model, prompts)
    N = len(prompts)
    scores = np.zeros((N, graph.n_edges), dtype=np.float32)
    embs = {L: np.zeros((N, D_MODEL), dtype=np.float32) for L in emb_layers}
    m_clean = np.zeros(N, dtype=np.float32); m_corr = np.zeros(N, dtype=np.float32)
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p["tokens"]), []).append(i)
    done = 0
    for L, idxs in sorted(by_len.items()):
        for s in range(0, len(idxs), batch_size):
            ids = idxs[s:s + batch_size]
            batch = [prompts[i] for i in ids]
            tc = torch.tensor([p["tokens"] for p in batch], device=device)
            tx = torch.tensor([p["tokens_corrupt"] for p in batch], device=device)
            sc, em, mc, mx = eap_batch(model, tc, tx, metric_factory(batch), graph, emb_layers)
            scores[ids] = sc.numpy()
            for l in emb_layers:
                embs[l][ids] = em[l].numpy()
            m_clean[ids] = mc.numpy(); m_corr[ids] = mx.numpy()
            done += len(ids)
            if done % (batch_size * 50) == 0 or done == N:
                log(f"  eap {done}/{N}")
    return scores, embs, m_clean, m_corr
