"""E4-interface RNNs for the HD task. Nodes = hidden units; mean ablation replaces a unit's activation by its
measure-batch mean (over batch and time) at every step. Exposes node_means / node_scores / edge_scores /
scores_under exactly as E4's discovery pipeline needs (the rescoring hook is routed through scores_under)."""
import math, numpy as np, torch, torch.nn as nn
from e4lib.tasks import metric
import e4lib.discovery as _disc

_orig_scores_under = _disc._scores_under
def _scores_under(model, tokens, targets, means, keep):
    if hasattr(model, "scores_under"): return model.scores_under(tokens, targets, means, keep)
    return _orig_scores_under(model, tokens, targets, means, keep)
_disc._scores_under = _scores_under      # adapter only; the pipeline logic in e4lib.discovery is untouched

class HDRNN(nn.Module):
    def __init__(self, N=148, K=16, cell="rnn", seed=0, leak=0.5):
        super().__init__(); torch.manual_seed(seed)
        self.N, self.K, self.cell, self.arch, self.leak = N, K, cell, "rnn", leak; self.n_nodes = N
        self.node_names = [f"h{i}" for i in range(N)]; self.node_layer = [0] * N; self.node_kind = ["neuron"] * N
        self.w_in = nn.Linear(3, N if cell == "rnn" else 3 * N)
        self.w_rec = nn.Linear(N, N if cell == "rnn" else 3 * N, bias=cell != "rnn")
        with torch.no_grad():
            g = 1.0 if cell == "rnn" else 1.0
            for blk in range(1 if cell == "rnn" else 3):
                nn.init.orthogonal_(self.w_rec.weight[blk * N:(blk + 1) * N], gain=g)
        self.readout = nn.Linear(N, K)

    @property
    def W_rec(self):
        """(N,N) recurrent matrix in v<-u orientation (row v, col u). GRU: the candidate-state block."""
        W = self.w_rec.weight.detach(); return W[-self.N:] if self.cell == "gru" else W

    def weight_graph(self):
        """E[u,v] = weight u->v (E4 edge convention)."""
        return self.W_rec.T.cpu().numpy().copy()

    def _step(self, h, x, keep, means):
        if self.cell == "rnn":
            pre = self.w_rec(h) + self.w_in(x); hn = torch.tanh(pre)
            hn = (1 - self.leak) * h + self.leak * hn                 # leaky (CTRNN-discretisation) step, SPEC amendment 1
        else:
            gi = self.w_in(x); gh = self.w_rec(h); N = self.N
            r = torch.sigmoid(gi[:, :N] + gh[:, :N]); z = torch.sigmoid(gi[:, N:2 * N] + gh[:, N:2 * N])
            pre = gi[:, 2 * N:] + r * gh[:, 2 * N:]; n = torch.tanh(pre); hn = (1 - z) * n + z * h
        if keep is not None:
            k = keep.float(); hn = hn * k + means * (1 - k)
        return hn, pre

    def forward(self, x, keep=None, means=None, record=False):
        B, T, _ = x.shape; h = x.new_zeros(B, self.N); hs, pres = [], []
        if means is not None and not torch.is_tensor(means): means = torch.stack(list(means)).reshape(-1)
        for t in range(T):
            h, pre = self._step(h, x[:, t], keep, means); hs.append(h); pres.append(pre)
        H = torch.stack(hs, 1)
        if record: self._hs, self._pres, self._H = hs, pres, H
        return self.readout(H)

    @torch.no_grad()
    def node_means(self, x):
        self.forward(x, record=True); return self._H.mean((0, 1)).detach()   # (N,)

    def _grads(self, x, targets, means, keep=None):
        self.zero_grad(set_to_none=True)
        logits = self.forward(x, keep=keep, means=means, record=True); m = metric(logits, targets)
        gh, gp = torch.autograd.grad(m, self._hs + self._pres, allow_unused=True)[: len(self._hs)], None
        return m, gh

    def node_scores(self, x, targets, means):
        return self.scores_under(x, targets, means, None)

    def scores_under(self, x, targets, means, keep):
        """Node attribution patching (mean corruption) with the ablated set held at means: sum_t (mu - h_t) . dm/dh_t."""
        self.zero_grad(set_to_none=True)
        logits = self.forward(x, keep=keep, means=means, record=True); m = metric(logits, targets)
        gh = torch.autograd.grad(m, self._hs, allow_unused=True)
        sc = torch.zeros(self.N)
        for h, g in zip(self._hs, gh):
            if g is not None: sc += ((means - h) * g).sum(0).detach()
        return sc

    def edge_scores(self, x, targets, means):
        """EAP: E[u,v] = sum_{b,t} W[v,u] (mu_u - h_{t,u}) dm/dpre_{t+1,v}. Returns (E, node_scores)."""
        self.zero_grad(set_to_none=True)
        logits = self.forward(x, record=True); m = metric(logits, targets)
        grads = torch.autograd.grad(m, self._hs + self._pres, allow_unused=True)
        gh, gp = grads[: len(self._hs)], grads[len(self._hs):]
        D = torch.stack([means - h for h in self._hs[:-1]], 1).reshape(-1, self.N)          # (B(T-1), N) delta_u
        G = torch.stack([g if g is not None else torch.zeros_like(self._pres[0]) for g in gp[1:]], 1)
        G = G[..., -self.N:].reshape(-1, self.N)                                              # (B(T-1), N) dm/dpre_v (GRU: candidate pre)
        W = self.W_rec                                                                       # (v,u)
        E = (W.T * (D.T @ G)).detach()                                                       # E[u,v]
        sc = torch.zeros(self.N)
        for h, g in zip(self._hs, gh):
            if g is not None: sc += ((means - h) * g).sum(0).detach()
        return E, sc

def planted_ring(N=148, J0=-0.3, J1=1.0, width=None):
    """Classical bump-attractor profile: W[i,j] = J0 + J1 cos(theta_i - theta_j) (symmetric, u->v). Anchor, not trained."""
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    W = J0 + J1 * np.cos(th[:, None] - th[None, :]); np.fill_diagonal(W, 0); return W
