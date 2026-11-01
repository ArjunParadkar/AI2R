"""Algorithmic tasks: modadd, sort, copy. Every task yields (tokens, targets) LongTensors.
tokens: (B, T) ; targets: (B, S) with S output slots; read_pos: positions to read logits at."""
import numpy as np, torch

P_MOD = 23
L_SEQ = 4
V_DIG = 8

class Task:
    def __init__(self, name, split_seed=0, random_labels=False):
        self.name = name
        rng = np.random.default_rng(split_seed)
        if name == "modadd":
            a, b = np.meshgrid(np.arange(P_MOD), np.arange(P_MOD), indexing="ij")
            X = np.stack([a.ravel(), b.ravel(), np.full(P_MOD * P_MOD, P_MOD)], 1)  # '=' token = P_MOD
            Y = ((a + b) % P_MOD).ravel()[:, None]
            self.vocab, self.n_classes, self.read_pos = P_MOD + 1, P_MOD, [2]
            self.chance = 1.0 / P_MOD
        elif name in ("sort", "copy"):
            X = np.array(np.meshgrid(*[np.arange(V_DIG)] * L_SEQ, indexing="ij")).reshape(L_SEQ, -1).T
            Y = np.sort(X, 1) if name == "sort" else X.copy()
            self.vocab, self.n_classes, self.read_pos = V_DIG, V_DIG, list(range(L_SEQ))
            self.chance = 1.0 / V_DIG
        else:
            raise ValueError(name)
        if random_labels:  # fixed random relabelling of every input (memorisation control)
            Y = rng.integers(0, self.n_classes, size=Y.shape)
        n = len(X); perm = rng.permutation(n)
        if random_labels or name == "modadd":  # train == eval: the function on its whole domain (SPEC amend. 1)
            tr, ev = perm, perm
        else:
            k = int(0.7 * n); tr, ev = perm[:k], perm[k:]
        self.X = torch.tensor(X, dtype=torch.long); self.Y = torch.tensor(Y, dtype=torch.long)
        self.train_idx, self.eval_idx = torch.tensor(tr), torch.tensor(ev)
        self.T, self.S = self.X.shape[1], self.Y.shape[1]

    def batch(self, n, seed, split="eval"):
        idx = self.train_idx if split == "train" else self.eval_idx
        g = torch.Generator().manual_seed(int(seed))
        sel = idx[torch.randperm(len(idx), generator=g)[:n]]
        return self.X[sel], self.Y[sel]

    def all(self, split="eval"):
        idx = self.train_idx if split == "train" else self.eval_idx
        return self.X[idx], self.Y[idx]

def metric(logits, targets):
    """Mean log-prob of the correct class over batch and output slots. logits (B,S,C)."""
    lp = torch.log_softmax(logits, -1)
    return lp.gather(-1, targets.unsqueeze(-1)).mean()

def accuracy(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()
