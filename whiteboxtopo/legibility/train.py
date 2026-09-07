import time, torch, torch.nn.functional as F
from .cluster_loss import clusterability_loss, block_mask_
from .tasks import accuracy

ARMS = {
    "none": {}, "wd0.1": dict(wd=0.1), "l1_1e-4": dict(l1=1e-4), "l1_1e-3": dict(l1=1e-3),
    "drop0.1": dict(dropout=0.1), "drop0.3": dict(dropout=0.3), "clus5": dict(clus=5.0), "clus40": dict(clus=40.0),
}

def train(model, task, wd=0.0, l1=0.0, clus=0.0, steps=4000, lr=1e-3, bs=256, seed=0, device="cpu", target_acc=0.99, log=None):
    torch.manual_seed(seed); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr, Ytr = task.all("train"); Xev, Yev = task.all("eval")
    Xtr, Ytr, Xev, Yev = Xtr.to(device), Ytr.to(device), Xev.to(device), Yev.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed); t0 = time.time(); acc = 0.0
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, len(Xtr), (min(bs, len(Xtr)),), generator=g)
        logits = model(Xtr[idx]); loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Ytr[idx].reshape(-1))
        if l1: loss = loss + l1 * sum(p.abs().sum() for p in model.parameters() if p.dim() >= 2)
        if clus: loss = loss + clus * clusterability_loss(model.clusterable)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0:
            model.eval()
            with torch.no_grad(): acc = accuracy(model(Xev), Yev)
            if log: log(f"  step {step} loss {loss.item():.3f} eval_acc {acc:.3f}")
            if acc >= target_acc: break
    model.eval()
    with torch.no_grad(): acc = accuracy(model(Xev), Yev); tr_acc = accuracy(model(Xtr[:2048]), Ytr[:2048])
    return dict(steps=step, eval_acc=acc, train_acc=tr_acc, train_time=time.time() - t0)
