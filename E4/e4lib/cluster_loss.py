"""Clusterability loss — re-implementation of Golechha, Chaudhary, Velja, Abate & Schoots,
'Studying Cross-cluster Modularity in Neural Networks' (arXiv 2502.02470, 2025), Sec. 3:
C(W) = sum_{i,j in same cluster} W_ij^2 / sum_{i,j} W_ij^2 with k fixed contiguous equal-size
clusters on each axis; training loss L = L_CE - lambda * C. We use lambda * (1 - C), identical
gradient. NOT our idea; cited as theirs."""
import torch

def _labels(n, k):
    return torch.arange(n) * k // n

def clusterability(W, k=4):
    W2 = W.pow(2)
    rl, cl = _labels(W.shape[0], k).to(W.device), _labels(W.shape[1], k).to(W.device)
    same = (rl[:, None] == cl[None, :]).float()
    return (W2 * same).sum() / (W2.sum() + 1e-12)

def clusterability_loss(mats, k=4):
    return torch.stack([1 - clusterability(W, k) for W in mats]).mean()

@torch.no_grad()
def block_mask_(model, k=4):
    """Zero every cross-cluster entry of every >=2-D weight (untrained high-Q control)."""
    for p in model.parameters():
        if p.dim() < 2: continue
        W = p.view(-1, p.shape[-2], p.shape[-1]) if p.dim() == 3 else p.unsqueeze(0)
        rl, cl = _labels(W.shape[1], k).to(p.device), _labels(W.shape[2], k).to(p.device)
        same = (rl[:, None] == cl[None, :]).float()
        W.mul_(same)
