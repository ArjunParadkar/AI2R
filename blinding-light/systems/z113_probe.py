"""Probe extraction for the Z113 transformer: the two layers we open, and the maps out of them.

layer 'resid_mid'  -- residual stream at the read-off position after attention, before the MLP.
                      psi(h) = (h + MLP(h)) W_U is genuinely nonlinear, so the pullback metric,
                      the curvature and the Lipschitz field all vary over the manifold.  This is
                      the primary layer.
layer 'resid_post' -- after the MLP.  psi(h) = h W_U is linear, so g is a single fixed quadratic
                      form; topology still means something there, the geometry does not.  Kept
                      as a control, and reported as such.
"""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, P, NULL, EQ, all_pairs
from whiteboxtopo.extract import jacobians, logits_of


def load(ckpt, device='cuda', seed=0):
    m = Z113Transformer(seed=seed).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def acts(model, toks, layer='resid_mid', device='cuda'):
    t = torch.as_tensor(toks, device=device)
    with torch.no_grad():
        x = model.embed(t)
        x = x + model.attn(x)
        mid = x[:, -1, :]
        post = mid + model.mlp(mid)
    return (mid if layer == 'resid_mid' else post).cpu().numpy()


def psi_of(model, layer='resid_mid'):
    if layer == 'resid_mid':
        return lambda h: (h + model.mlp(h)) @ model.W_U
    return lambda h: h @ model.W_U


def probe_set(n=2000, seed=0, hint=NULL, hint_values=None):
    """A fixed probe: n random (a,b) pairs, with the hint token held at NULL unless asked."""
    rng = np.random.default_rng(seed)
    a, b, c = all_pairs()
    sel = rng.choice(len(a), size=n, replace=False)
    h = np.full(n, hint) if hint_values is None else hint_values[sel]
    toks = np.stack([a[sel], b[sel], h, np.full(n, EQ)], 1)
    return dict(toks=toks, a=a[sel], b=b[sel], c=c[sel], idx=sel, hint=h)


def layer_data(model, probe, layer='resid_mid', device='cuda'):
    from whiteboxtopo.problems import LayerData
    X = acts(model, probe['toks'], layer, device)
    psi = psi_of(model, layer)
    J = jacobians(psi, X, device=device)
    lg = logits_of(psi, X, device=device)
    variables = dict(a=probe['a'], b=probe['b'], c=probe['c'],
                     a_plus_b=(probe['a'] + probe['b']) % P,
                     a_minus_b=(probe['a'] - probe['b']) % P)
    return LayerData(name=layer, X=X.astype(np.float32), J=J.astype(np.float32), logits=lg,
                     labels=probe['c'], variables=variables, modular_period=P), psi
