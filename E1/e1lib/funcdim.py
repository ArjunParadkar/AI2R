"""Functional dimension (Grigsby, Lindsey, Meyerhoff, Wu 2022): rank of the Jacobian of the
realisation map theta -> (f_theta(x_j))_j at the trained parameters."""
import numpy as np
import torch


def functional_dimension(model, rng, m_factor=4, rel_tol=1e-6, box=(-1.0, 2.0)):
    n_params = model.n_params()
    m = m_factor * n_params
    X = torch.tensor(rng.uniform(box[0], box[1], size=(m, model.n_in)), dtype=torch.float64)
    Ws = [torch.tensor(W, requires_grad=True) for W in model.Ws]
    bs = [torch.tensor(b, requires_grad=True) for b in model.bs]

    def f(*params):
        Wp, bp = params[: len(Ws)], params[len(Ws):]
        h = X
        for W, b in zip(Wp[:-1], bp[:-1]):
            h = torch.relu(h @ W + b)
        return (h @ Wp[-1] + bp[-1])[:, 0]

    J = torch.autograd.functional.jacobian(f, tuple(Ws + bs))
    J = torch.cat([j.reshape(m, -1) for j in J], dim=1).numpy()
    s = np.linalg.svd(J, compute_uv=False)
    rank = int((s > rel_tol * s[0]).sum()) if s[0] > 0 else 0
    n_hidden = sum(model.widths)
    # dead units: never active on the probe inputs
    h = X.numpy()
    dead, dead_dirs = 0, 0
    for l, (W, b) in enumerate(zip(model.Ws[:-1], model.bs[:-1])):
        pre = h @ W + b
        nd = int(((pre > 0).sum(0) == 0).sum())
        dead += nd
        # a dead unit zeroes the Jacobian on its fan_in+1 incoming params and its fan_out
        # outgoing params; its scaling direction is already in `expected`, so the extra is fan_in+fan_out
        dead_dirs += nd * (W.shape[0] + model.Ws[l + 1].shape[1])
        h = np.maximum(pre, 0.0)
    expected = n_params - n_hidden
    deficit = expected - rank
    return {"n_params": n_params, "rank": rank, "expected_rank": expected,
            "deficit": deficit, "dead_units": dead,
            "unexplained_deficit": max(deficit - dead_dirs, 0)}
