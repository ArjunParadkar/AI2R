"""Z113 modular-addition transformer (Nanda grokking recipe), with optional planted shortcut.

Sequence is always [a, b, h, =] of length 4.
  a, b in 0..112          -- the operands
  h in 0..112 or NULL     -- the planted hint token (NULL for the clean system)
  =  the read-off position; logits at the final position predict c = (a+b) mod 113.

The hint is the nuisance variable z for P3: in the shortcut system it equals the answer on a
random half of the training set, so a circular coordinate tracking h leaks c.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 113
NULL = P        # 113 -- "no hint"
EQ = P + 1      # 114 -- the "=" token
VOCAB = P + 2
NCTX = 4


class Z113Transformer(nn.Module):
    """1 layer, d=128, 4 heads, ReLU MLP, no LayerNorm. Attention written out by hand:
    we never touch SDPA, whose EFFICIENT_ATTENTION backward is wrong on this GPU."""

    def __init__(self, d_model=128, n_heads=4, d_mlp=512, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        def par(*shape, scale):
            return nn.Parameter(torch.randn(*shape, generator=g) * scale)
        s = d_model ** -0.5
        self.W_E = par(VOCAB, d_model, scale=s)
        self.W_pos = par(NCTX, d_model, scale=s)
        self.W_Q = par(n_heads, d_model, self.d_head, scale=s)
        self.W_K = par(n_heads, d_model, self.d_head, scale=s)
        self.W_V = par(n_heads, d_model, self.d_head, scale=s)
        self.W_O = par(n_heads, self.d_head, d_model, scale=s)
        self.W_in = par(d_model, d_mlp, scale=s)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = par(d_mlp, d_model, scale=s)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = par(d_model, P, scale=s)

    def embed(self, toks):
        return self.W_E[toks] + self.W_pos[None, : toks.shape[1]]

    def attn(self, x):
        q = torch.einsum('bpd,hdk->bphk', x, self.W_Q)
        k = torch.einsum('bpd,hdk->bphk', x, self.W_K)
        v = torch.einsum('bpd,hdk->bphk', x, self.W_V)
        scores = torch.einsum('bphk,bqhk->bhpq', q, k) / (self.d_head ** 0.5)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, -1e30)
        a = scores.softmax(-1)
        z = torch.einsum('bhpq,bqhk->bphk', a, v)
        return torch.einsum('bphk,hkd->bpd', z, self.W_O)

    def mlp(self, x):
        return F.relu(x @ self.W_in + self.b_in) @ self.W_out + self.b_out

    def resid_final(self, toks):
        """Residual stream at the read-off position -- this is h_ell, the layer we open."""
        x = self.embed(toks)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x[:, -1, :]

    def head(self, h):
        """psi_ell: the remaining network from the read-off residual to logits."""
        return h @ self.W_U

    def forward(self, toks):
        return self.head(self.resid_final(toks))


def all_pairs():
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing='ij')
    a, b = a.ravel(), b.ravel()
    return a, b, (a + b) % P


def build_data(seed, mode, frac=0.3, n_leak_residues=40):
    """Returns train/test tensors. mode in {clean, shortcut, scrambled, decoy}.

    The planted shortcut follows the spec: the hint token leaks the answer for a SUBSET OF
    RESIDUES.  For c in S (|S| = n_leak_residues) every training example carries h = c; for
    c outside S the hint is NULL.  So z = the hint is a Z113-valued (circular) nuisance
    variable that is perfectly predictive of the label on S and absent elsewhere.
    """
    rng = np.random.default_rng(seed)
    a, b, c = all_pairs()
    n = len(a)
    perm = rng.permutation(n)
    ntr = int(frac * n)
    tr, te = perm[:ntr], perm[ntr:]

    h = np.full(n, NULL, dtype=np.int64)
    leak_residues = np.array([], dtype=np.int64)
    if mode == 'shortcut':
        leak_residues = np.sort(rng.choice(P, size=n_leak_residues, replace=False))
        is_leaky = np.isin(c, leak_residues)
        h[is_leaky] = c[is_leaky]          # the hint IS the answer, on those residues

    if mode == 'decoy':
        # the harder P3 null: a hint token is present exactly as often as in 'shortcut', but it
        # is drawn uniformly and independently of the answer -- z is there and carries nothing.
        has = rng.random(n) < n_leak_residues / P
        h[has] = rng.integers(0, P, size=int(has.sum()))

    if mode == 'scrambled_decoy':
        # Run 4: the three-slot memoriser -- uninformative hint tokens exactly as in 'decoy', AND random labels
        has = rng.random(n) < n_leak_residues / P
        h[has] = rng.integers(0, P, size=int(has.sum()))

    y = c.copy()
    if mode == 'aonly':
        y = a.copy()                       # Run 4 null that can fail: the answer is a mod 113; b is irrelevant BY DESIGN
    if mode == 'scrambled_decoy':
        y = rng.integers(0, P, size=n)
    if mode == 'scrambled':
        y = rng.integers(0, P, size=n)     # random labels: nothing to understand

    def toks(idx, hv):
        return np.stack([a[idx], b[idx], hv[idx], np.full(len(idx), EQ)], 1)

    out = dict(
        train_toks=toks(tr, h), train_y=y[tr],
        test_toks=toks(te, h), test_y=y[te],
        train_idx=tr, test_idx=te, a=a, b=b, c=c,
        leak_residues=leak_residues,
        train_leaky=np.isin(c[tr], leak_residues) if len(leak_residues) else np.zeros(len(tr), bool),
        test_leaky=np.isin(c[te], leak_residues) if len(leak_residues) else np.zeros(len(te), bool),
    )
    if mode == 'shortcut':
        # ablated: the hint removed wherever it appears (this is the ground-truth onset probe)
        out['abl_toks'] = toks(te, np.full(n, NULL))
        # decorrelated test set: hint uniform over Z113 u {NULL}, independent of the answer
        hd = rng.integers(0, P + 1, size=n)
        out['dec_toks'] = toks(te, hd)
        out['dec_hint'] = hd[te]
        # worst group: the hint is present and wrong
        out['dec_worst'] = (hd[te] != NULL) & (hd[te] != y[te])
    return out


def run(mode, seed, steps, outdir, frac=0.3, ckpt_every_pct=2.0, lr=1e-3, wd=1.0, device='cuda'):
    os.makedirs(outdir, exist_ok=True)
    d = build_data(seed, mode, frac=frac)
    T = lambda x: torch.as_tensor(x, device=device)
    xtr, ytr = T(d['train_toks']), T(d['train_y'])
    xte, yte = T(d['test_toks']), T(d['test_y'])
    extra = {}
    for key in ('dec', 'abl'):
        if f'{key}_toks' in d:
            extra[key] = T(d[f'{key}_toks'])

    torch.manual_seed(seed)
    model = Z113Transformer(seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    ckpt_every = max(1, int(steps * ckpt_every_pct / 100))

    hist, t0 = [], time.time()
    for step in range(steps + 1):
        if step % ckpt_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                ltr = model(xtr); lte = model(xte)
                rec = dict(
                    step=step,
                    train_loss=F.cross_entropy(ltr, ytr).item(),
                    test_loss=F.cross_entropy(lte, yte).item(),
                    train_acc=(ltr.argmax(-1) == ytr).float().mean().item(),
                    test_acc=(lte.argmax(-1) == yte).float().mean().item(),
                )
                if 'dec' in extra:
                    ld = model(extra['dec']); la = model(extra['abl'])
                    rec['dec_acc'] = (ld.argmax(-1) == yte).float().mean().item()
                    rec['abl_acc'] = (la.argmax(-1) == yte).float().mean().item()
                    w = T(d['dec_worst'])
                    rec['dec_worst_acc'] = (ld.argmax(-1) == yte)[w].float().mean().item()
                    # ground-truth shortcut onset: hint present vs hint ablated, on TRAIN leaky rows
                    lk = T(d['train_leaky'])
                    if lk.any():
                        ltr_l = ltr.argmax(-1)[lk] == ytr[lk]
                        abl_tr = xtr.clone(); abl_tr[:, 2] = NULL
                        la_tr = model(abl_tr).argmax(-1)[lk] == ytr[lk]
                        rec['leaky_acc'] = ltr_l.float().mean().item()
                        rec['leaky_acc_ablated'] = la_tr.float().mean().item()
                        rec['shortcut_reliance'] = rec['leaky_acc'] - rec['leaky_acc_ablated']
            hist.append(rec)
            torch.save(model.state_dict(), f'{outdir}/ckpt_{step:06d}.pt')
            model.train()
            if step % (ckpt_every * 10) == 0:
                print(f"[{mode} s{seed}] {step:6d} train {rec['train_acc']:.3f} test {rec['test_acc']:.3f}"
                      + (f" abl {rec.get('abl_acc', float('nan')):.3f}" if 'abl_acc' in rec else ''), flush=True)
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xtr), ytr).backward()
        opt.step()

    meta = dict(mode=mode, seed=seed, steps=steps, ckpt_every=ckpt_every, lr=lr, wd=wd,
                wall_s=time.time() - t0, P=P, NULL=NULL, EQ=EQ)
    json.dump(dict(meta=meta, history=hist), open(f'{outdir}/history.json', 'w'), indent=1)
    np.savez(f'{outdir}/data.npz', **{k: v for k, v in d.items() if isinstance(v, np.ndarray)})
    print(f"[{mode} s{seed}] done in {meta['wall_s']:.0f}s -> {outdir}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['clean', 'shortcut', 'scrambled', 'decoy', 'scrambled_decoy', 'aonly'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--frac', type=float, default=0.3)
    a = ap.parse_args()
    run(a.mode, a.seed, a.steps, a.outdir, frac=a.frac)
