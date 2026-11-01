"""V7 — modularity drift over training on public checkpoints (HPCC): Pythia-410M and Pythia-1B, all 154 revisions
(step0 .. step143000), V2's structure measures (Leiden Q + degree-preserving rewired null, min-fill treewidth bound,
participation-ratio sparsity, at edge densities 0.01 / 0.05 / 0.1) on a head/MLP-level EAP attribution graph around the
mid layer, from a FIXED 2 000-token probe (the K1 OpenWebText-10k slice, first 16 sequences x 125 tokens) so that anyone
can verify the numbers. NOVELTY_BOARD.md #6: cite 2502.02470 (Pythia-70m clusterability) and Filan 2003.04881.

Graph: nodes = attention heads and the MLP of layers [mid-2, mid+2] (5 x (H + 1) nodes); EAP edge score u -> v (u in an
earlier layer) = sum_t (mean_u - a_u(t)) . dL/d x_in(v)(t), mean corruption over the probe, metric = next-token
log-prob (Pythia's parallel residual means every node of a layer reads the same layer input, so one gradient per layer).
Usage:  python mejiro_ardan.py --model EleutherAI/pythia-410m --out results/v7 --resume [--revisions step0 step1000 ...]
Checkpoint: one CSV row per (model, revision, density); --resume skips finished revisions. ~1 min/revision on an A100."""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "K_darkmatter"))
from whiteboxtopo import graphs
from whiteboxtopo.legibility.graph import symmetrise, threshold_density
DENS = (0.01, 0.05, 0.1)
REVISIONS = ["step0"] + [f"step{2**i}" for i in range(0, 10)] + [f"step{k}" for k in range(1000, 144000, 1000)]


class _PerHeadQKV(torch.nn.Module):
    """Recompute the fused ``query_key_value`` output head by head, each head reading its OWN clone of the block input
    through its own application of ``input_layernorm``, so that dL/dx_in is head-specific (V AMENDMENTS #V7-1 /
    SPEC_O3 #O3-2). Numerically identical to the fused projection — the rows of ``query_key_value`` are head-major
    ``[q_h|k_h|v_h]``, so concatenating the per-head row blocks reproduces it exactly, at the same matmul flops (the
    LayerNorm is recomputed H times, which is negligible). ``eps`` is a zero leaf added to the block input on head h's
    path only, so ``d lp / d eps_h`` IS ``dL/dx_in(head h)`` in residual-stream coordinates."""

    def __init__(self, lin, ln, H, dh, store):
        super().__init__(); self.lin, self.ln, self.H, self.dh, self.store = lin, ln, H, dh, store

    def forward(self, h_ln):                    # h_ln = input_layernorm(x); ignored, recomputed per head
        x = self.store["x"]; W = self.lin.weight; b = getattr(self.lin, "bias", None)
        parts = []
        for h in range(self.H):
            eps = torch.zeros_like(x, requires_grad=True); self.store["eps"][f"h{h}"] = eps
            r = slice(h * 3 * self.dh, (h + 1) * 3 * self.dh)
            parts.append(torch.nn.functional.linear(self.ln(x + eps), W[r], None if b is None else b[r]))
        return torch.cat(parts, -1)


def eap_graph_per_node(model, T, layers, dev):
    """Per-target-node EAP (V AMENDMENTS #V7-1): identical to ``eap_graph`` except that every consumer node v reads its
    own cloned copy of the layer input, so ``dL/dx_in(v)`` is head/MLP-specific instead of one gradient per layer. The
    node set, the edge-allowance rule (u -> v only for strictly later layers, Pythia's parallel residual), the mean
    corruption and the metric are unchanged; only the gradient is. Returns (E, nodes) like ``eap_graph``."""
    cfg = model.config; H = cfg.num_attention_heads; d = cfg.hidden_size; dh = d // H
    nodes = [(l, f"h{h}") for l in layers for h in range(H)] + [(l, "mlp") for l in layers]; idx = {n: i for i, n in enumerate(nodes)}
    acts = {}; stores = {l: {"x": None, "eps": {}} for l in layers}; handles = []; orig = {}
    for l in layers:
        L = model.gpt_neox.layers[l]
        handles.append(L.register_forward_pre_hook(lambda mod, inp, l=l: stores[l].__setitem__("x", inp[0])))   # KEPT in the graph
        def _mlp_pre(mod, inp, l=l):
            eps = torch.zeros_like(inp[0], requires_grad=True); stores[l]["eps"]["mlp"] = eps
            return (inp[0] + eps,) + tuple(inp[1:])                                    # a pre-hook's return REPLACES the inputs
        handles.append(L.post_attention_layernorm.register_forward_pre_hook(_mlp_pre))
        handles.append(L.attention.dense.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__((l, "z"), inp[0].detach())))
        handles.append(L.mlp.register_forward_hook(lambda mod, inp, out, l=l: acts.__setitem__((l, "mlp"), out.detach())))
        orig[l] = L.attention.query_key_value
        L.attention.query_key_value = _PerHeadQKV(orig[l], L.input_layernorm, H, dh, stores[l])
    try:
        out = model(T); logits = out.logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, T[:, 1:, None]).squeeze(-1).sum()
        keys = [(l, n) for l in layers for n in ([f"h{h}" for h in range(H)] + ["mlp"])]
        leaves = [stores[l]["eps"][n] for l, n in keys]
        gs = torch.autograd.grad(lp, leaves, retain_graph=False, allow_unused=True)
        gnode = {k: (g.detach().float() if g is not None else None) for k, g in zip(keys, gs)}
    finally:
        for h_ in handles: h_.remove()
        for l in layers: model.gpt_neox.layers[l].attention.query_key_value = orig[l]
    E = np.zeros((len(nodes), len(nodes))); contrib = {}
    for l in layers:
        Wo = model.gpt_neox.layers[l].attention.dense.weight.detach().float()
        z = acts[(l, "z")].float()
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); contrib[(l, f"h{h}")] = z[..., sl] @ Wo[:, sl].T
        contrib[(l, "mlp")] = acts[(l, "mlp")].float()
    for (lu, nu), a_u in contrib.items():
        delta = a_u.mean((0, 1), keepdim=True) - a_u
        for lv in layers:
            if lv <= lu: continue
            for nv in [f"h{h}" for h in range(H)] + ["mlp"]:
                g = gnode.get((lv, nv))
                if g is not None: E[idx[(lu, nu)], idx[(lv, nv)]] = float((delta * g).sum())
    model.zero_grad(set_to_none=True); return E, nodes


def eap_graph(model, T, layers, dev):
    """Signed directed EAP score matrix over nodes (layer, head|mlp) for the layers listed.

    PRE-REGISTERED instrument (V7 SPEC). Known defect, recorded in V AMENDMENTS #V7-1 and SPEC_O3 #O3-2: one gradient
    is read per *layer*, so every consumer node in a layer receives the SAME score from a given source — on pythia-70m
    at step143000 the symmetrised matrix carries 91 distinct edge magnitudes among 990 node pairs, and the density
    threshold's tie-breaking makes Q move by CV 0.57 under exact renamings. ``eap_graph_per_node`` is the repair; this
    function is kept unchanged because the V7 rows were pre-registered with it."""
    cfg = model.config; H = cfg.num_attention_heads; d = cfg.hidden_size; dh = d // H
    nodes = [(l, f"h{h}") for l in layers for h in range(H)] + [(l, "mlp") for l in layers]; idx = {n: i for i, n in enumerate(nodes)}
    acts = {}; grads = {}; handles = []
    for l in layers:
        L = model.gpt_neox.layers[l]
        def _pre(mod, inp, l=l):      # a pre-hook's return value REPLACES the layer inputs, so it must return None (laptop smoke 2026-09-10)
            inp[0].register_hook(lambda g, l=l: grads.__setitem__(l, g.detach()))
        handles.append(L.register_forward_pre_hook(_pre))
        handles.append(L.attention.dense.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__((l, "z"), inp[0].detach())))
        handles.append(L.mlp.register_forward_hook(lambda mod, inp, out, l=l: acts.__setitem__((l, "mlp"), out.detach())))
    out = model(T); logits = out.logits[:, :-1].float(); lp = torch.log_softmax(logits, -1).gather(-1, T[:, 1:, None]).squeeze(-1).sum()
    lp.backward()
    for h_ in handles: h_.remove()
    # per-head contributions to the residual: z_h @ W_O[:, h-slice]^T
    E = np.zeros((len(nodes), len(nodes)))
    contrib = {}
    for l in layers:
        Wo = model.gpt_neox.layers[l].attention.dense.weight.detach().float()   # (d, d): out = z @ Wo^T
        z = acts[(l, "z")].float()
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh); contrib[(l, f"h{h}")] = z[..., sl] @ Wo[:, sl].T
        contrib[(l, "mlp")] = acts[(l, "mlp")].float()
    for (lu, nu), a_u in contrib.items():
        delta = a_u.mean((0, 1), keepdim=True) - a_u                                   # mean corruption
        for lv in layers:
            if lv <= lu: continue
            g = grads[lv].float(); s = float((delta * g).sum())
            for nv in [f"h{h}" for h in range(H)] + ["mlp"]: E[idx[(lu, nu)], idx[(lv, nv)]] = s
    model.zero_grad(set_to_none=True); return E, nodes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="EleutherAI/pythia-410m"); ap.add_argument("--revisions", nargs="*", default=None)
    ap.add_argument("--out", default="results/v7"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--n-seq", type=int, default=16); ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-node", action="store_true", help="use the per-target-node EAP gradient (V AMENDMENTS #V7-1 repair). "
                                                            "DEFAULT OFF: the V7 rows are pre-registered with the per-layer gradient.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = a.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import eishin_flash
    tok = AutoTokenizer.from_pretrained(a.model); eishin_flash.CACHE = a.out; toks = eishin_flash.load_tokens(tok)[: a.n_seq, :126]; T = torch.tensor(toks, device=dev)
    tag = a.model.split("/")[-1] + ("_pernode" if a.per_node else ""); csv_p = os.path.join(a.out, f"V7_{tag}.csv"); rows = pd.read_csv(csv_p).to_dict("records") if (a.resume and os.path.exists(csv_p)) else []
    done = {r["revision"] for r in rows}
    for rev in (a.revisions or REVISIONS):
        if rev in done: continue
        t0 = time.time()
        try: model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, torch_dtype=torch.float32).to(dev).eval()
        except Exception as e: print("skip", rev, e, flush=True); continue
        mid = model.config.num_hidden_layers // 2; layers = list(range(mid - 2, mid + 3))
        E, nodes = (eap_graph_per_node if a.per_node else eap_graph)(model, T, layers, dev); A_full = symmetrise(E)
        for dens in DENS:
            A = threshold_density(A_full, dens); Q, memb = graphs.modularity(A); q0, sd = graphs.rewired_null(A, n=20)
            rows.append(dict(model=a.model, revision=rev, step=int(rev[4:]), density=dens, per_node=int(a.per_node), n_nodes=len(nodes), n_edges=int((A != 0).sum() // 2), Q=float(Q), Q_null=float(q0), Q_null_sd=float(sd),
                             Q_excess=float(Q - q0), n_communities=int(len(np.unique(memb))), treewidth_ub=int(graphs.treewidth_ub(A)), sparsity=float(graphs.sparsity(A_full)), wall_s=round(time.time() - t0)))
        pd.DataFrame(rows).to_csv(csv_p, index=False); print(rows[-1], flush=True); del model; torch.cuda.empty_cache()
    print("done", csv_p)


if __name__ == "__main__":
    main()
