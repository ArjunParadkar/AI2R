"""Finite (Jacobian-free) g-length of the straight segment between two activations, measured at the output.

    d_g(h, h') = sum_{k<K} || psi(h_{k+1}) - psi(h_k) ||,   h_k = h + (k/K)(h' - h)

psi is the network from the layer to the logits.  For a linear psi this equals ||psi(h') - psi(h)||; in general it is >= that secant and it
converges to the pullback length  int_0^1 || J(h(t)) (h'-h) || dt  as K grows.  Run 5 (SPEC5.md, block T) uses it in place of the linearised
quotient  || J(h) (h'-h) ||, whose two Run-4 artefacts were both "infinitesimal misses finite".  Neighbouring work: LieGG (Moskalev et al. 2022)."""
import torch


def finite_g_length(psi, h0, h1, K=16, chunk=4096):
    """psi: batched map (N, *shape) -> (N, out).  h0, h1: (n, *shape).  Returns (n,) lengths."""
    n = len(h0); t = torch.linspace(0, 1, K + 1, dtype=h0.dtype, device=h0.device).reshape(K + 1, *([1] * h0.dim()))
    pts = (h0[None] + t * (h1 - h0)[None]).reshape((K + 1) * n, *h0.shape[1:])
    with torch.no_grad():
        out = torch.cat([psi(pts[i:i + chunk]) for i in range(0, len(pts), chunk)]).reshape(K + 1, n, -1)
    return (out[1:] - out[:-1]).norm(dim=-1).sum(0)


def finite_q(psi, h, hT, hother, K=16):
    """q = mean_x d_g(h(x), h(Tx)) / mean_{x,x'} d_g(h(x), h(x'))."""
    return float(finite_g_length(psi, h, hT, K).mean() / finite_g_length(psi, h, hother, K).mean())
