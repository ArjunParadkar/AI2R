"""F7 — toy transformers on Boolean tasks: training, masked forward, exhaustive circuit
enumeration over {heads} ∪ {MLP units}, and gauge-canonical keys (SPEC §F7)."""
import hashlib, time
import numpy as np, torch
import torch.nn.functional as F
from whiteboxtopo.symmetry import ToyTransformer
from .functions import truth_table
from .minimal import minimal_ids_fast

EQ = 2   # '=' token


def make_tokens(fn):
    X, y = truth_table(fn)
    tok = np.concatenate([X.astype(np.int64), np.full((len(X), 1), EQ)], 1)
    return tok, y.astype(np.int64)


def train_toy(fn, n_layers, seed, device="cuda", max_steps=5000, lr=3e-3, margin=0.5):
    tok, y = make_tokens(fn)
    d_mlp = 16 if n_layers == 1 else 8
    model = ToyTransformer(vocab=3, seq_len=tok.shape[1], d_model=32, n_heads=4, n_layers=n_layers, d_mlp=d_mlp, n_out=2, seed=seed).to(device)
    T = torch.tensor(tok, device=device); Y = torch.tensor(y, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    conv, steps = False, max_steps
    for step in range(max_steps):
        opt.zero_grad(); out = model(T)[:, -1]; loss = F.cross_entropy(out, Y); loss.backward(); opt.step()
        with torch.no_grad():
            out = model(T)[:, -1]; diff = out[:, 1] - out[:, 0]
            if (((diff > 0).long() == Y).all() and ((diff * (2 * Y - 1)) >= margin).all()):
                conv, steps = True, step + 1; break
    return model.double().eval(), conv, steps


def node_layout(model):
    """Node ids: per layer l, heads 0..H-1 then MLP units 0..m-1."""
    per = model.H + model.m
    return per, model.L * per


@torch.no_grad()
def masked_forward(model, tok, keep, means, scheme):
    """tok (N, T) long; keep (M, n_nodes) float; means: dict with per-layer 'head' (T,H,dh) and 'mlp' (T,m)
    (means over inputs, positions kept). Returns logits (M, N, n_out) at the last position."""
    M = keep.shape[0]; N, T = tok.shape; per = model.H + model.m
    x = (model.wte[tok] + model.wpe[:T][None])[None].expand(M, N, T, model.d)
    for l in range(model.L):
        h = F.layer_norm(x, (model.d,), model.ln1_g[l], model.ln1_b[l], 1e-5)
        q = torch.einsum("mbtd,dhk->mbthk", h, model.Wq[l]) + model.bq[l]
        k = torch.einsum("mbtd,dhk->mbthk", h, model.Wk[l]) + model.bk[l]
        v = torch.einsum("mbtd,dhk->mbthk", h, model.Wv[l]) + model.bv[l]
        att = torch.einsum("mbthk,mbshk->mbhts", q, k) / np.sqrt(model.dh)
        att = att.masked_fill(~model.mask[:T, :T], float("-inf")).softmax(-1)
        z = torch.einsum("mbhts,mbshk->mbthk", att, v)                                  # (M,N,T,H,dh)
        km = keep[:, l * per: l * per + model.H][:, None, None, :, None]                # (M,1,1,H,1)
        rep = means["head"][l][None, None] if scheme == "mean" else torch.zeros_like(z[:1, :1])
        z = z * km + rep * (1 - km)
        x = x + torch.einsum("mbthk,hkd->mbtd", z, model.Wo[l]) + model.bo[l]
        h2 = F.layer_norm(x, (model.d,), model.ln2_g[l], model.ln2_b[l], 1e-5)
        a = torch.relu(h2 @ model.W1[l] + model.b1[l])                                   # (M,N,T,m)
        ku = keep[:, l * per + model.H:(l + 1) * per][:, None, None, :]
        rep = means["mlp"][l][None, None] if scheme == "mean" else torch.zeros_like(a[:1, :1])
        a = a * ku + rep * (1 - ku)
        x = x + a @ model.W2[l] + model.b2[l]
    out = F.layer_norm(x[:, :, -1], (model.d,), model.lnf_g, model.lnf_b, 1e-5) @ model.Wu + model.bu
    return out


@torch.no_grad()
def node_means(model, tok):
    _, cache = model(tok, return_cache=True)
    return {"head": [z.mean(0) for z in cache["heads"]], "mlp": [a.mean(0) for a in cache["mlp"]]}


@torch.no_grad()
def enumerate_valid_tf(model, tok, y, scheme, chunk=1 << 13, time_cap=None):
    t0 = time.time(); per, n_nodes = node_layout(model); total = 1 << n_nodes
    dev = model.wte.device; T = torch.tensor(tok, device=dev); Y = torch.tensor(y, device=dev)
    means = node_means(model, T); bits = torch.arange(n_nodes, device=dev)
    valid = []
    for start in range(0, total, chunk):
        if time_cap and time.time() - t0 > time_cap: return None
        ids = torch.arange(start, min(start + chunk, total), device=dev)
        keep = ((ids[:, None] >> bits[None, :]) & 1).double()
        out = masked_forward(model, T, keep, means, scheme)
        pred = (out[..., 1] > out[..., 0]).long()
        ok = (pred == Y[None]).all(1)
        valid.append(ids[ok].cpu().numpy())
    return np.concatenate(valid)


def _fold_ln(g, b, W, bias):
    """Fold LayerNorm affine (g, b) into the following linear map: W~ = diag(g) W, b~ = bias + b W."""
    return g[:, None] * W, bias + b @ W


@torch.no_grad()
def canonical_key_tf(model, mid, eps):
    """Gauge-canonical hash of the sub-implementation of circuit ``mid`` (node bitmask):
    heads -> rotation-invariant, LN-fold-invariant bilinear forms (W~Q W~K^T, W~V W_O, folded biases),
    sorted; MLP units -> LN-folded, per-unit scale-normalised incoming (w, b) with the norm pushed
    into W2, lexicographically sorted. Quantised at ``eps`` and hashed."""
    per, n_nodes = node_layout(model)
    h = hashlib.blake2b(digest_size=16)
    parts = []
    for l in range(model.L):
        heads = [hh for hh in range(model.H) if (mid >> (l * per + hh)) & 1]
        units = [u for u in range(model.m) if (mid >> (l * per + model.H + u)) & 1]
        g1, b1 = model.ln1_g[l], model.ln1_b[l]
        hk = []
        for hh in heads:
            Wq, bq = _fold_ln(g1, b1, model.Wq[l][:, hh], model.bq[l][hh]); Wk, bk = _fold_ln(g1, b1, model.Wk[l][:, hh], model.bk[l][hh])
            Wv, bv = _fold_ln(g1, b1, model.Wv[l][:, hh], model.bv[l][hh]); Wo = model.Wo[l][hh]
            QK = Wq @ Wk.T; qb = Wq @ bk; kb = Wk @ bq; bb = bq @ bk            # q_i . k_j = x_i QK x_j^T + x_i qb + kb . x_j + bb
            OV = Wv @ Wo; ob = bv @ Wo
            hk.append(np.concatenate([np.round(t.cpu().numpy().ravel() / eps) for t in (QK, qb, kb, bb[None], OV, ob)]).astype(np.int64))
        hk.sort(key=lambda a: a.tobytes())
        g2, b2 = model.ln2_g[l], model.ln2_b[l]
        W1, bb1 = _fold_ln(g2, b2, model.W1[l], model.b1[l])
        uk = []
        for u in units:
            w, b = W1[:, u], bb1[u]; nrm = torch.sqrt((w ** 2).sum() + b ** 2); nrm = nrm if nrm > 0 else torch.tensor(1.0, dtype=w.dtype)
            wo = model.W2[l][u] * nrm
            uk.append(np.concatenate([np.round(t.cpu().numpy().ravel() / eps) for t in (w / nrm, (b / nrm)[None], wo)]).astype(np.int64))
        uk.sort(key=lambda a: a.tobytes())
        parts.append((len(heads), len(units)))
        for a in hk + uk: h.update(a.tobytes())
        h.update(bytes(str((len(heads), len(units))), "ascii"))
    Wu, bu = _fold_ln(model.lnf_g, model.lnf_b, model.Wu, model.bu)
    for t in (Wu, bu, model.wte, model.wpe):
        h.update(np.round(t.cpu().numpy().ravel() / eps).astype(np.int64).tobytes())
    return h.hexdigest(), parts
