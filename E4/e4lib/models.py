"""Node-instrumented models. A node is a hidden neuron (MLP) or an attention head / MLP neuron
(transformer). Every model exposes:
  n_nodes, node_layer (int per node), node_kind ('neuron'|'head'), node_names
  forward(tokens, keep=None, means=None, record=False) -> logits (B,S,C)
      keep: bool tensor (n_nodes,) — False nodes are mean-ablated using `means`
      record=True stores per-node output tensors + input-gradient handles for attribution
  node_means(tokens) -> list of per-node mean outputs (batch-averaged, position kept)
  edge_scores(tokens, targets, means) -> (n_nodes, n_nodes) EAP scores (u -> v), and node scores
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .tasks import metric

class _Base(nn.Module):
    def out_logits(self, tokens, keep=None, means=None, record=False):
        return self.forward(tokens, keep=keep, means=means, record=record)

    @torch.no_grad()
    def node_means(self, tokens):
        self.forward(tokens, record=True)
        return [o.mean(0 if self.arch == "mlp" else 1, keepdim=True).detach() for o in self._node_out]

    def node_scores(self, tokens, targets, means):
        """Node-level attribution patching: sum (mean - out) * dm/dout."""
        self.zero_grad(set_to_none=True)
        logits = self.forward(tokens, record=True)
        m = metric(logits, targets)
        outs = self._node_out
        grads = torch.autograd.grad(m, outs, allow_unused=True)
        sc = []
        for o, g, mu in zip(outs, grads, means):
            if g is None: sc.append(0.0); continue
            sc.append(((mu - o) * g).sum().item())
        return torch.tensor(sc)

class MLP(_Base):
    arch = "mlp"
    def __init__(self, task, width=64, dropout=0.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task; self.in_dim = task.vocab * task.T
        self.S, self.C = task.S, task.n_classes
        self.l1 = nn.Linear(self.in_dim, width); self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, self.S * self.C); self.drop = nn.Dropout(dropout)
        self.n_nodes = 2 * width; self.width = width
        self.node_layer = [0] * width + [1] * width
        self.node_kind = ["neuron"] * self.n_nodes
        self.node_names = [f"L{l}.n{i}" for l in range(2) for i in range(width)]
        self.clusterable = [self.l1.weight, self.l2.weight, self.l3.weight]

    def forward(self, tokens, keep=None, means=None, record=False):
        x = F.one_hot(tokens, self.task.vocab).float().flatten(1)
        W = self.width
        a1 = F.relu(self.l1(x))                         # per-node outputs are the scalars a1[:, i]
        if keep is not None:
            k1 = keep[:W].to(a1.device).float(); a1 = a1 * k1 + torch.cat(means[:W], 1) * (1 - k1)
        a1 = self.drop(a1)
        pre2 = self.l2(a1)
        if record and pre2.requires_grad: pre2.retain_grad()
        a2 = F.relu(pre2)
        if keep is not None:
            k2 = keep[W:].to(a2.device).float(); a2 = a2 * k2 + torch.cat(means[W:], 1) * (1 - k2)
        a2 = self.drop(a2)
        if record:
            self._node_out = [a1[:, i:i + 1] for i in range(W)] + [a2[:, i:i + 1] for i in range(W)]
            self._pre2 = pre2; self._a1 = a1
        return self.l3(a2).view(-1, self.S, self.C)

    def edge_scores(self, tokens, targets, means):
        self.zero_grad(set_to_none=True)
        logits = self.forward(tokens, record=True); m = metric(logits, targets)
        outs = self._node_out; W = self.width
        grads = torch.autograd.grad(m, outs + [self._pre2], allow_unused=True)
        g_pre2 = grads[-1]                                    # (B, W) dm/dpre_v for layer-2 nodes
        d1 = torch.cat([mu - o for o, mu in zip(outs[:W], means[:W])], 1)  # (B, W) delta of L1 outputs
        E = torch.zeros(self.n_nodes, self.n_nodes)
        E[:W, W:] = (self.l2.weight.T * (d1.T @ g_pre2)).detach().cpu()   # E[u,v] = W2[v,u] sum_b d1_u g_v
        node = torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item()
                             for o, g, mu in zip(outs, grads[:-1], means)])
        return E, node

class TinyTF(_Base):
    arch = "tf"
    def __init__(self, task, n_layers=2, d=32, heads=4, mlp=64, dropout=0.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task; self.S, self.C = task.S, task.n_classes
        self.d, self.H, self.dh, self.M, self.nL = d, heads, d // heads, mlp, n_layers
        self.emb = nn.Embedding(task.vocab, d); self.pos = nn.Embedding(task.T, d)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        s = 1 / math.sqrt(d)
        self.Wq = nn.ParameterList([nn.Parameter(torch.randn(heads, d, self.dh) * s) for _ in range(n_layers)])
        self.Wk = nn.ParameterList([nn.Parameter(torch.randn(heads, d, self.dh) * s) for _ in range(n_layers)])
        self.Wv = nn.ParameterList([nn.Parameter(torch.randn(heads, d, self.dh) * s) for _ in range(n_layers)])
        self.Wo = nn.ParameterList([nn.Parameter(torch.randn(heads, self.dh, d) * s) for _ in range(n_layers)])
        self.Win = nn.ParameterList([nn.Parameter(torch.randn(d, mlp) * s) for _ in range(n_layers)])
        self.bin = nn.ParameterList([nn.Parameter(torch.zeros(mlp)) for _ in range(n_layers)])
        self.Wout = nn.ParameterList([nn.Parameter(torch.randn(mlp, d) / math.sqrt(mlp)) for _ in range(n_layers)])
        self.bout = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d); self.unemb = nn.Linear(d, task.n_classes)
        self.drop = nn.Dropout(dropout)
        self.read_pos = torch.tensor(task.read_pos)
        per = heads + mlp; self.n_nodes = n_layers * per; self.per = per
        self.node_layer, self.node_kind, self.node_names = [], [], []
        for l in range(n_layers):
            self.node_layer += [2 * l] * heads + [2 * l + 1] * mlp
            self.node_kind += ["head"] * heads + ["neuron"] * mlp
            self.node_names += [f"L{l}.h{h}" for h in range(heads)] + [f"L{l}.n{i}" for i in range(mlp)]
        self.clusterable = list(self.Win) + list(self.Wout)

    def forward(self, tokens, keep=None, means=None, record=False):
        B, T = tokens.shape
        x = self.emb(tokens) + self.pos(torch.arange(T, device=tokens.device))
        outs, ins = [], []
        for l in range(self.nL):
            base = l * self.per
            xh = x.unsqueeze(0).expand(self.H, B, T, self.d).contiguous()   # per-head input copy
            if record: xh.requires_grad_(True); xh.retain_grad()
            h = self.ln1[l](xh)
            q = torch.einsum("hbtd,hde->hbte", h, self.Wq[l]); k = torch.einsum("hbtd,hde->hbte", h, self.Wk[l])
            v = torch.einsum("hbtd,hde->hbte", h, self.Wv[l])
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), -1)
            o = torch.einsum("hbte,hed->hbtd", att @ v, self.Wo[l])            # (H,B,T,d) head outputs
            if keep is not None:
                for hh in range(self.H):
                    if not keep[base + hh]: o = o.clone(); o[hh] = means[base + hh].expand(B, T, self.d)
            o = self.drop(o)
            x = x + o.sum(0)
            xc = x.unsqueeze(2).expand(B, T, self.M, self.d).contiguous()     # per-neuron input copy
            if record: xc.requires_grad_(True); xc.retain_grad()
            pre = (self.ln2[l](xc) * self.Win[l].T).sum(-1) + self.bin[l]      # (B,T,M)
            act = F.relu(pre)
            if keep is not None:
                km = keep[base + self.H: base + self.per].to(act.device).float()
                mu = torch.stack([means[base + self.H + i][..., 0] for i in range(self.M)], -1)  # (1,T,M)
                act = act * km + mu * (1 - km)
            act = self.drop(act)
            x = x + act @ self.Wout[l] + self.bout[l]
            if record:
                outs += [o[hh] for hh in range(self.H)] + [act[..., i:i + 1] for i in range(self.M)]
                ins.append((xh, xc))
        if record: self._node_out, self._ins = outs, ins
        logits = self.unemb(self.lnf(x[:, self.read_pos]))
        return logits

    def edge_scores(self, tokens, targets, means):
        self.zero_grad(set_to_none=True)
        logits = self.forward(tokens, record=True); m = metric(logits, targets)
        outs = self._node_out
        grads = torch.autograd.grad(m, outs + [t for pair in self._ins for t in pair], allow_unused=True)
        gout, gin = grads[:len(outs)], grads[len(outs):]
        B, T = tokens.shape; D = B * T * self.d
        # delta of each node's contribution to the residual stream, flattened to (N, B*T*d)
        deltas = []
        for i, (o, mu) in enumerate(zip(outs, means)):
            l, j = divmod(i, self.per)
            if j < self.H: deltas.append((mu.expand(B, T, self.d) - o).reshape(-1))
            else: deltas.append(((mu - o) * self.Wout[l][j - self.H]).reshape(-1))
        Dm = torch.stack(deltas).detach()                                     # (N, D)
        # gradient at each node's input copy, flattened
        gl = []
        for l in range(self.nL):
            gh, gc = gin[2 * l], gin[2 * l + 1]
            gl += [gh[hh].reshape(-1) for hh in range(self.H)] + [gc[:, :, i, :].reshape(-1) for i in range(self.M)]
        G = torch.stack(gl).detach()                                          # (N, D)
        E = (Dm @ G.T).cpu()
        layer = torch.tensor(self.node_layer)
        E = E * (layer[:, None] < layer[None, :]).float()                    # u must be strictly upstream of v
        node = torch.tensor([0.0 if g is None else ((mu - o) * g).sum().item()
                             for o, g, mu in zip(outs, gout, means)])
        return E, node

def build_model(arch, task, dropout=0.0, seed=0):
    if arch == "mlp": return MLP(task, dropout=dropout, seed=seed)
    if arch == "tf2": return TinyTF(task, n_layers=2, dropout=dropout, seed=seed)
    if arch == "tf3": return TinyTF(task, n_layers=3, dropout=dropout, seed=seed)
    if arch == "tf4": return TinyTF(task, n_layers=4, dropout=dropout, seed=seed)
    raise ValueError(arch)
