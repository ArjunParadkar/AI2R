"""Exhaustive circuit enumeration on GPU (F2 large cells): validity of every 2^n_hidden subset
under zero- and mean-ablation, in chunks of 2^20 masks. Returns sorted valid mask ids."""
import numpy as np
import torch


def enumerate_valid_torch(model, X, y, scheme, device="cuda", chunk=1 << 20, time_cap=None):
    import time
    t0 = time.time()
    widths = model.widths; n_hidden = sum(widths); total = 1 << n_hidden
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Ws = [torch.tensor(W, dtype=torch.float32, device=device) for W in model.Ws]
    bs = [torch.tensor(b, dtype=torch.float32, device=device) for b in model.bs]
    means = [torch.tensor(a.mean(0), dtype=torch.float32, device=device) for a in model.hidden_acts(X)]
    yt = torch.tensor(y, dtype=torch.int8, device=device)
    bits = torch.arange(n_hidden, device=device)
    off = np.cumsum([0] + list(widths))
    valid = []
    for start in range(0, total, chunk):
        if time_cap is not None and time.time() - t0 > time_cap:
            return None
        ids = torch.arange(start, min(start + chunk, total), device=device, dtype=torch.int64)
        masks = ((ids[:, None] >> bits[None, :]) & 1).to(torch.float32)            # (B, n_hidden)
        h = Xt[None].expand(len(ids), -1, -1)
        for l in range(len(widths)):
            h = torch.relu(h @ Ws[l] + bs[l])
            m = masks[:, off[l]:off[l + 1]][:, None, :]
            h = h * m + (0.0 if scheme == "zero" else means[l][None, None, :]) * (1 - m)
        logits = (h @ Ws[-1] + bs[-1])[..., 0]
        ok = ((logits > 0).to(torch.int8) == yt[None, :]).all(1)
        valid.append(ids[ok].cpu().numpy())
    return np.concatenate(valid) if valid else np.zeros(0, dtype=np.int64)
