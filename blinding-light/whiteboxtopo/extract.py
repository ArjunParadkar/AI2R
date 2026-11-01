"""Getting (activations, Jacobians, logits) off a model, fast.

The Jacobian of the remaining network psi_l is what defines the pullback metric, so it is
computed for every probe point.  torch.func.jacrev under vmap does this in one batched pass.
"""
import numpy as np
import torch


def jacobians(psi, X, batch=256, device='cuda', dtype=torch.float32):
    """J[i] = d psi / d h at X[i].  psi maps (B, d_l) -> (B, d_out) and must be batch-wise
    independent (no cross-sample mixing), which every 'rest of the network' map is."""
    from torch.func import jacrev, vmap
    X = torch.as_tensor(X, device=device, dtype=dtype)
    single = lambda h: psi(h[None])[0]
    jac = vmap(jacrev(single))
    out = []
    for s in range(0, len(X), batch):
        with torch.no_grad():
            pass
        out.append(jac(X[s:s + batch]).detach().cpu().numpy())
    return np.concatenate(out, 0)


def logits_of(psi, X, batch=1024, device='cuda', dtype=torch.float32):
    X = torch.as_tensor(X, device=device, dtype=dtype)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(psi(X[s:s + batch]).cpu().numpy())
    return np.concatenate(out, 0)
