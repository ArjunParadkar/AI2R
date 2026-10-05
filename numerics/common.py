"""Manufactured test objects with analytic ground truth."""
import numpy as np, torch
T64 = torch.float64


def mlp_weights(sizes, seed, scale=1.0, dtype=T64):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(o, i, generator=g, dtype=dtype) * scale / i ** 0.5,
             torch.randn(o, generator=g, dtype=dtype) * 0.1) for i, o in zip(sizes[:-1], sizes[1:])]


def tanh_mlp(Ws):
    def f(x):
        for W, b in Ws[:-1]: x = torch.tanh(W @ x + b)
        W, b = Ws[-1]; return W @ x + b
    return f


def tanh_mlp_J(Ws, x):
    """hand back-propagated Jacobian (independent of autograd)."""
    J = torch.eye(len(x), dtype=x.dtype); a = x
    for W, b in Ws[:-1]:
        z = W @ a + b; a = torch.tanh(z)
        J = torch.diag(1 - a ** 2) @ W @ J
    W, b = Ws[-1]; return W @ J


def poly_map(x):
    return torch.stack([x[0] * x[1] + torch.sin(x[2]), x[1] ** 2 - x[3] * x[4],
                        torch.exp(0.3 * x[0]) * torch.cos(x[4]), x[2] * x[3] + torch.tanh(x[1])])


def poly_map_J(x):
    x0, x1, x2, x3, x4 = [float(v) for v in x]
    J = torch.zeros(4, 5, dtype=x.dtype)
    J[0] = torch.tensor([x1, x0, np.cos(x2), 0, 0], dtype=x.dtype)
    J[1] = torch.tensor([0, 2 * x1, 0, -x4, -x3], dtype=x.dtype)
    J[2] = torch.tensor([0.3 * np.exp(0.3 * x0) * np.cos(x4), 0, 0, 0, -np.exp(0.3 * x0) * np.sin(x4)], dtype=x.dtype)
    J[3] = torch.tensor([0, 1 - np.tanh(x1) ** 2, x3, x2, 0], dtype=x.dtype)
    return J


def relu_mlp(Ws):
    def f(x):
        for W, b in Ws[:-1]: x = torch.relu(W @ x + b)
        W, b = Ws[-1]; return W @ x + b
    return f


def rewrite(Ws, seed):
    """permutation + positive rescaling of every hidden layer; returns new weights and per-layer M = D P."""
    g = torch.Generator().manual_seed(seed)
    Ws = [(W.clone(), b.clone()) for W, b in Ws]; Ms = []
    for l in range(len(Ws) - 1):
        h = Ws[l][0].shape[0]
        P = torch.eye(h, dtype=T64)[torch.randperm(h, generator=g)]
        D = torch.diag(torch.exp(torch.empty(h, dtype=T64).uniform_(np.log(0.1), np.log(10.0), generator=g)))
        M = D @ P
        Ws[l] = (M @ Ws[l][0], M @ Ws[l][1])
        Ws[l + 1] = (Ws[l + 1][0] @ torch.linalg.inv(M), Ws[l + 1][1]); Ms.append(M)
    return Ws, Ms


def sphere_embed(p):          # (theta, phi) -> R^3
    th, ph = p[0], p[1]
    return torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)])


def halton_np(n, lo, hi, seed=0):
    from whiteboxtopo.numerics.killing import halton
    return halton(n, lo, hi, seed)
