"""whiteboxtopo.metric -- the one Riemannian metric everything in run 3 is read from.

g(x) = J(x)^T J(x), the pullback of the Euclidean metric on the output of f to its input.  J is
d_out x d_in with d_out small, so g is never formed: its nonzero spectrum is the spectrum of the
d_out x d_out Gram matrix J J^T (exact -- rank g <= d_out), and g v = J^T (J v).

SPEC3 S1.
"""
import torch

__all__ = ['pullback_input', 'gram_spectrum', 'g_matvec']


def _jac_backward(f, xb):
    """d_out backward passes through one forward pass.  Valid when samples in a batch do not
    interact (a network in eval mode).  Returns (B, d_out, d_in)."""
    xb = xb.detach().clone().requires_grad_(True)
    out = f(xb)
    out = out.reshape(out.shape[0], -1)
    rows = []
    for k in range(out.shape[1]):
        g, = torch.autograd.grad(out[:, k].sum(), xb, retain_graph=(k < out.shape[1] - 1))
        rows.append(g.reshape(g.shape[0], -1))
    return torch.stack(rows, 1).detach()


def _jac_func(f, xb):
    from torch.func import jacrev, vmap

    def single(x):
        return f(x[None]).reshape(-1)
    J = vmap(jacrev(single))(xb)
    return J.reshape(J.shape[0], J.shape[1], -1).detach()


def pullback_input(f, x, batch=32, method='backward'):
    """Jacobian of f at each sample.  x: (N, *shape) -> J: (N, d_out, d_in), d_in = prod(shape).

    method 'backward': d_out backward passes per batch (robust for conv nets / batch-norm in eval mode);
    method 'func': torch.func vmap(jacrev).  On CUDA out-of-memory the batch is halved and retried.
    """
    fn = _jac_backward if method == 'backward' else _jac_func
    outs, s, b = [], 0, max(1, int(batch))
    while s < len(x):
        try:
            outs.append(fn(f, x[s:s + b]))
            s += b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if b == 1:
                raise
            b = max(1, b // 2)
    return torch.cat(outs, 0)


def gram_spectrum(J, return_vectors=False):
    """Exact nonzero spectrum of g = J^T J from the small Gram matrix J J^T.

    J: (..., d_out, d_in).  Returns eigenvalues in DESCENDING order, shape (..., d_out); with
    return_vectors also U (..., d_out, d_out), columns the eigenvectors of J J^T in the same order
    (the matching unit eigenvectors of g are J^T u / sqrt(lambda))."""
    G = J @ J.transpose(-1, -2)
    lam, U = torch.linalg.eigh(G.double())
    lam = lam.flip(-1).clamp_min(0.0)
    U = U.flip(-1)
    if return_vectors:
        return lam.to(J.dtype), U.to(J.dtype)
    return lam.to(J.dtype)


def g_matvec(J, v):
    """g v = J^T (J v) without forming g.  J: (..., d_out, d_in), v: (..., d_in) or (..., k, d_in)."""
    if v.dim() == J.dim() - 1:
        return (J.transpose(-1, -2) @ (J @ v.unsqueeze(-1))).squeeze(-1)
    return (v @ J.transpose(-1, -2)) @ J
