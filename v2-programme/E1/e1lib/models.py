"""Plain-numpy MLP container + torch training. Weights stored as lists:
W[l] shape (fan_in, fan_out), b[l] shape (fan_out,). Hidden layers 0..d-1, output layer d."""
import numpy as np
import torch


class MLP:
    def __init__(self, Ws, bs):
        self.Ws = [np.asarray(W, dtype=np.float64) for W in Ws]
        self.bs = [np.asarray(b, dtype=np.float64) for b in bs]

    @property
    def depth(self):  # number of hidden layers
        return len(self.Ws) - 1

    @property
    def widths(self):
        return [W.shape[1] for W in self.Ws[:-1]]

    @property
    def n_in(self):
        return self.Ws[0].shape[0]

    def hidden_acts(self, X):
        """Post-ReLU activations per hidden layer in the full network."""
        h = X
        acts = []
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(h @ W + b, 0.0)
            acts.append(h)
        return acts

    def logits(self, X):
        h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        return (h @ self.Ws[-1] + self.bs[-1])[:, 0]

    def predict(self, X):
        return (self.logits(X) > 0).astype(np.int8)

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.Ws, self.bs))

    def copy(self):
        return MLP([W.copy() for W in self.Ws], [b.copy() for b in self.bs])

    def save(self, path):
        np.savez(path, **{f"W{i}": W for i, W in enumerate(self.Ws)}, **{f"b{i}": b for i, b in enumerate(self.bs)})

    @staticmethod
    def load(path):
        z = np.load(path)
        L = len([k for k in z.files if k.startswith("W")])
        return MLP([z[f"W{i}"] for i in range(L)], [z[f"b{i}"] for i in range(L)])


def train_mlp(X, y, depth, width, seed, lr=1e-2, max_steps=20000, margin=0.5):
    """Full-batch Adam on BCE; stop at zero error with |logit|>=margin everywhere.
    Returns (MLP, converged: bool, steps)."""
    torch.manual_seed(seed)
    n = X.shape[1]
    dims = [n] + [width] * depth + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    converged, steps = False, max_steps
    for step in range(max_steps):
        opt.zero_grad()
        out = net(Xt)[:, 0]
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            out = net(Xt)[:, 0]
            if ((out > 0).float() == yt).all() and (out.abs() >= margin).all():
                converged, steps = True, step + 1
                break
    Ws = [m.weight.detach().numpy().T for m in net if isinstance(m, torch.nn.Linear)]
    bs = [m.bias.detach().numpy() for m in net if isinstance(m, torch.nn.Linear)]
    return MLP(Ws, bs), converged, steps


def lookup_table_mlp(X, y, depth=1):
    """One-hot lookup network: one hidden unit per input pattern (fires 1 exactly on
    that pattern: w = 2x-1, b = 1 - |x|₁), output = sum over true patterns - 0.5.
    depth>1 pads extra identity-like layers? No — keep depth 1 (SPEC control 3)."""
    n = X.shape[1]
    m = X.shape[0]
    W1 = np.zeros((n, m))
    b1 = np.zeros(m)
    for j, x in enumerate(X):
        W1[:, j] = 2 * x - 1
        b1[j] = 1 - x.sum()
    W2 = (y.astype(np.float64) * 2 - 1).reshape(m, 1)   # +1 for true patterns, -1 for false
    b2 = np.zeros(1)
    return MLP([W1, W2], [b1, b2])
