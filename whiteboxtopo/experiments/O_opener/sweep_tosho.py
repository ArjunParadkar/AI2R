#!/usr/bin/env python3
"""O1.1 Sweep Tosho — train the T1 (provable-tier) specimens per SPEC_O.md.

  python sweep_tosho.py grok  [--conditions clean planted memorize] [--seeds 0 1 2 3 4] [--epochs 30000]
  python sweep_tosho.py ring  [--seeds 0..9] [--steps 4000]

Resumable: a specimen with results/ckpt/<name>/final.pt is skipped. Writes results/train_log.csv (append),
results/curves/<name>.csv, checkpoints every 1 % (grok) / 2 % (ring) of the budget. results/specimens.csv and
results/O1_1_excluded.csv are rebuilt from the final.pt files by o11_diagnose.py (AMENDMENTS #2).
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from whiteboxtopo.symmetry.transformer import ToyTransformer  # noqa: E402

RES = os.environ.get("O_RES", os.path.join(HERE, "results")); CK = os.path.join(RES, "ckpt"); CURVES = os.path.join(RES, "curves")
for d in (RES, CK, CURVES): os.makedirs(d, exist_ok=True)
P = 113; N_SHORT = 8; EQ = P; SHORT0 = P + 1   # '=' token, first shortcut token
ONSET_FRAC = 0.10   # planted data onset (AMENDMENTS #4; SPEC_O had 0.30)
DECAY_FRAC = 0.10   # cosine LR decay to 0 over the last 10 % of the budget (AMENDMENTS #4)
RECIPE = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIELDS = ["name", "kind", "condition", "seed", "steps", "train_acc", "test_acc", "kappa", "ang_err", "included", "reason", "wall"]


def _append(path, row, fields=FIELDS):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); new and w.writeheader(); w.writerow(row)


def fourier_kappa(E: np.ndarray, p: int = P, tau: float = 0.05):
    """Fourier concentration of the token embedding (rows 0..p-1): fraction of DFT power (k=1..(p-1)//2, k and p-k folded)
    in frequencies whose share exceeds tau. Invariant to orthogonal rotations of the embedding columns. Mirrors
    whiteboxtopo.invariants.extractors.fourier_irreps (kept local so training does not depend on the module under construction)."""
    F = np.fft.fft(E[:p], axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    half = (p - 1) // 2
    folded = pw[1:half + 1] + pw[p - half:][::-1]
    share = folded / folded.sum()
    K = [int(k) for k in np.where(share > tau)[0] + 1]
    return float(share[share > tau].sum()), K


def fourier_kappa_f(L: np.ndarray, p: int = P, tau: float = 0.05):
    """Function-level Fourier concentration (AMENDMENTS #2). L: logit tensor (p, p, p) over (a, b, c). Centred over c, then the
    fraction of 3-D DFT power on the diagonal bins (k, k, -k) = the logit variance explained by a function of (a+b-c) mod p alone.
    Returns (kappa_f, K_f, share, n_eff): folded per-frequency shares s_k (k = 1..(p-1)//2), K_f = {k : s_k > tau}, n_eff = exp H(s).
    Invariant under the whole gauge group by construction (a function of the realisation map)."""
    L = np.asarray(L, dtype=np.float64).reshape(p, p, p)
    L = L - L.mean(-1, keepdims=True)
    pw = np.abs(np.fft.fftn(L)) ** 2
    k = np.arange(1, p); diag = pw[k, k, (-k) % p]
    half = (p - 1) // 2; folded = diag[:half] + diag[::-1][:half]
    tot = pw.sum(); kappa_f = float(folded.sum() / tot) if tot > 0 else 0.0
    share = folded / folded.sum() if folded.sum() > 0 else np.full(half, 1.0 / half)
    K = [int(i) + 1 for i in np.where(share > tau)[0]]
    n_eff = float(np.exp(-(share * np.log(share + 1e-300)).sum()))
    return kappa_f, K, share, n_eff


# ----------------------------------------------------------------------------------------------- grokked modular addition
def grok_data(seed: int, condition: str):
    rng = np.random.default_rng(seed)
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij"); a, b = a.ravel(), b.ravel()
    c = (a + b) % P
    if condition == "memorize":
        c = rng.permutation(c)                          # fixed random labels: nothing to generalise
    idx = rng.permutation(P * P); n_tr = int(0.3 * P * P)
    tr, te = idx[:n_tr], idx[n_tr:]
    return a, b, c, tr, te


def grok_tokens(a, b, s=None):
    cols = [a, b] + ([s] if s is not None else []) + [np.full_like(a, EQ)]
    return torch.as_tensor(np.stack(cols, 1), dtype=torch.long, device=DEV)


def marginal_logits(model, a, b, batch=4096):
    """Planted specimens (AMENDMENTS #3): logits at '=' averaged over the 8 shortcut values — the group-algorithm component."""
    acc = None
    with torch.no_grad():
        for s0 in range(SHORT0, SHORT0 + N_SHORT):
            s = np.full_like(a, s0); toks = grok_tokens(a, b, s); out = torch.cat([model(toks[i:i + batch])[:, -1] for i in range(0, len(toks), batch)])
            acc = out if acc is None else acc + out
    return (acc / N_SHORT).cpu().numpy()


def train_grok(condition: str, seed: int, epochs: int, ckpt_every: int):
    name = f"grok_{condition}_s{seed}"; out = os.path.join(CK, name)
    if os.path.exists(os.path.join(out, "final.pt")): print("skip", name, "(done)"); return
    os.makedirs(out, exist_ok=True); t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    a, b, c, tr, te = grok_data(seed, condition)
    planted = condition == "planted"
    seq_len = 4 if planted else 3
    vocab = P + 1 + (N_SHORT if planted else 0)
    model = ToyTransformer(vocab, seq_len, d_model=128, n_heads=4, n_layers=1, d_mlp=512, n_out=P, seed=seed).to(DEV)
    wd = 0.0 if condition == "memorize" else 1.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=wd)
    y = torch.as_tensor(c, dtype=torch.long, device=DEV)
    rng = np.random.default_rng(seed + 1000)
    s_leak = SHORT0 + ((a + b) % N_SHORT)
    onset = int(ONSET_FRAC * epochs); t_decay = int((1 - DECAY_FRAC) * epochs)
    lr_at = lambda ep: 1e-3 if ep < t_decay else 1e-3 * 0.5 * (1 + math.cos(math.pi * (ep - t_decay) / max(1, epochs - t_decay)))  # noqa: E731
    curve = open(os.path.join(CURVES, name + ".csv"), "w"); curve.write("epoch,train_loss,train_acc,test_acc,kappa,kappa_f,n_eff_f\n")
    def evaluate(ep):
        model.eval()
        with torch.no_grad():
            s = None
            if planted: s = s_leak if ep >= onset else rng.integers(SHORT0, SHORT0 + N_SHORT, size=a.shape)
            logits = model(grok_tokens(a, b, s))[:, -1]
            pred = logits.argmax(-1)
            tra = (pred[tr] == y[tr]).float().mean().item(); tea = (pred[te] == y[te]).float().mean().item()
        model.train(); return tra, tea, logits.cpu().numpy()
    for ep in range(epochs + 1):
        if ep % ckpt_every == 0 or ep == epochs:
            tra, tea, L = evaluate(ep)
            kappa, _ = fourier_kappa(model.wte.detach().cpu().numpy()); kappa_f, K_f, _, n_eff = fourier_kappa_f(L)
            torch.save({"state": model.state_dict(), "epoch": ep, "train_acc": tra, "test_acc": tea, "kappa": kappa, "kappa_f": kappa_f, "K_f": K_f, "n_eff_f": n_eff,
                        "cfg": dict(vocab=vocab, seq_len=seq_len, condition=condition, seed=seed, onset=onset if planted else None, recipe=RECIPE)},
                       os.path.join(out, f"step_{ep:06d}.pt"))
            curve.write(f"{ep},{loss.item() if ep else float('nan')},{tra:.4f},{tea:.4f},{kappa:.4f},{kappa_f:.4f},{n_eff:.2f}\n"); curve.flush()
            if ep % (ckpt_every * 20) == 0: print(f"{name} ep {ep} train {tra:.3f} test {tea:.3f} kappa_emb {kappa:.3f} kappa_f {kappa_f:.3f} n_eff {n_eff:.1f}", flush=True)
        if ep == epochs: break
        s = None
        if planted: s = s_leak if ep >= onset else rng.integers(SHORT0, SHORT0 + N_SHORT, size=a.shape)
        logits = model(grok_tokens(a[tr], b[tr], None if s is None else s[tr]))[:, -1]
        loss = nn.functional.cross_entropy(logits, y[tr])
        for g in opt.param_groups: g["lr"] = lr_at(ep)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    curve.close()
    tra, tea, L = evaluate(epochs); kappa, K = fourier_kappa(model.wte.detach().cpu().numpy()); kappa_f, K_f, _, n_eff = fourier_kappa_f(L)
    if planted: kappa_f_m, K_f_m, _, n_eff_m = fourier_kappa_f(marginal_logits(model, a, b)); ok, reason = inclusion_rule(condition, tea, kappa_f_m)
    else: ok, reason = inclusion_rule(condition, tea, kappa_f)
    torch.save({"state": model.state_dict(), "epoch": epochs, "train_acc": tra, "test_acc": tea, "kappa": kappa, "K": K,
                "kappa_f": kappa_f, "K_f": K_f, "n_eff_f": n_eff, "wall": time.time() - t0,
                "cfg": dict(vocab=vocab, seq_len=seq_len, condition=condition, seed=seed, onset=onset if planted else None, recipe=RECIPE)},
               os.path.join(out, "final.pt"))
    _append(os.path.join(RES, "train_log.csv"), dict(name=name, kind="grok", condition=condition, seed=seed, steps=epochs,
            train_acc=f"{tra:.4f}", test_acc=f"{tea:.4f}", kappa=f"{kappa:.4f}", ang_err="", included=int(ok), reason=reason,
            wall=f"{time.time() - t0:.0f}"))
    print(f"DONE {name} train {tra:.3f} test {tea:.3f} kappa_emb {kappa:.3f} kappa_f {kappa_f:.3f} n_eff {n_eff:.1f} K_f={K_f} included={ok} {reason}", flush=True)


def inclusion_rule(condition: str, test_acc: float, kappa_f: float):
    """O1.1 inclusion (SPEC_O + AMENDMENTS #2): memorising control iff test acc < 0.05; clean/planted iff test > 0.99 and kappa_f > 0.9."""
    if condition == "memorize":
        ok = test_acc < 0.05; return ok, "" if ok else f"memorising control generalised (test {test_acc:.3f})"
    ok = test_acc > 0.99 and kappa_f > 0.9
    return ok, "" if ok else ("not grokked (test %.3f)" % test_acc if test_acc <= 0.99 else "no Fourier structure (kappa_f %.3f)" % kappa_f)


# ----------------------------------------------------------------------------------------------- ring attractor RNN
class RingRNN(nn.Module):
    """Leaky ReLU RNN h_{t+1} = (1-a) h_t + a relu(W h_t + U u_t + b); u = (omega, cos th0 [t=0], sin th0 [t=0]); linear readout."""
    def __init__(self, n=128, alpha=0.1, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.n, self.alpha = n, alpha
        self.W = nn.Parameter(torch.randn(n, n) / math.sqrt(n)); self.U = nn.Parameter(torch.randn(3, n) * 0.5)
        self.b = nn.Parameter(torch.zeros(n)); self.R = nn.Parameter(torch.randn(n, 2) / math.sqrt(n)); self.rb = nn.Parameter(torch.zeros(2))
    def forward(self, u, return_states=False):
        B, T, _ = u.shape; h = torch.zeros(B, self.n, device=u.device); hs = []
        for t in range(T):
            h = (1 - self.alpha) * h + self.alpha * torch.relu(h @ self.W + u[:, t] @ self.U + self.b); hs.append(h)
        H = torch.stack(hs, 1); out = H @ self.R + self.rb
        return (out, H) if return_states else out


def ring_batch(rng, B, T, sigma=0.3):
    th0 = rng.uniform(0, 2 * np.pi, B)
    om = np.zeros((B, T)); x = rng.normal(0, sigma, B)
    for t in range(T): x = 0.9 * x + rng.normal(0, sigma * math.sqrt(1 - 0.81), B); om[:, t] = x   # OU velocity, stationary sd sigma
    th = th0[:, None] + np.cumsum(om, 1)
    u = np.zeros((B, T, 3)); u[:, :, 0] = om; u[:, 0, 1] = np.cos(th0); u[:, 0, 2] = np.sin(th0)
    y = np.stack([np.cos(th), np.sin(th)], -1)
    return (torch.as_tensor(u, dtype=torch.float32, device=DEV), torch.as_tensor(y, dtype=torch.float32, device=DEV), th)


def ang_error(out, th):
    pred = np.arctan2(out[..., 1], out[..., 0]); d = np.angle(np.exp(1j * (pred - th)))
    return float(np.abs(d).mean())


def train_ring(seed: int, steps: int, ckpt_every: int):
    name = f"ring_s{seed}"; out = os.path.join(CK, name)
    if os.path.exists(os.path.join(out, "final.pt")): print("skip", name, "(done)"); return
    os.makedirs(out, exist_ok=True); t0 = time.time()
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(10_000 + seed)
    model = RingRNN(seed=seed).to(DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    u_te, y_te, th_te = ring_batch(erng, 256, 200)
    curve = open(os.path.join(CURVES, name + ".csv"), "w"); curve.write("step,loss,ang_err200\n")
    def evaluate():
        model.eval()
        with torch.no_grad(): e = ang_error(model(u_te).cpu().numpy(), th_te)
        model.train(); return e
    loss = torch.tensor(float("nan"))
    for st in range(steps + 1):
        if st % ckpt_every == 0 or st == steps:
            e = evaluate(); torch.save({"state": model.state_dict(), "step": st, "ang_err": e, "cfg": dict(seed=seed)}, os.path.join(out, f"step_{st:06d}.pt"))
            curve.write(f"{st},{loss.item()},{e:.4f}\n"); curve.flush()
            if st % (ckpt_every * 10) == 0: print(f"{name} step {st} loss {loss.item():.4f} ang_err {e:.3f}", flush=True)
        if st == steps: break
        u, y, _ = ring_batch(rng, 128, 100)
        loss = ((model(u) - y) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    curve.close(); e = evaluate(); ok = e < 0.15; reason = "" if ok else f"angular error {e:.3f} >= 0.15"
    torch.save({"state": model.state_dict(), "step": steps, "ang_err": e, "wall": time.time() - t0, "cfg": dict(seed=seed)}, os.path.join(out, "final.pt"))
    _append(os.path.join(RES, "train_log.csv"), dict(name=name, kind="ring", condition="ring", seed=seed, steps=steps, train_acc="",
            test_acc="", kappa="", ang_err=f"{e:.4f}", included=int(ok), reason=reason, wall=f"{time.time() - t0:.0f}"))
    print(f"DONE {name} ang_err {e:.3f} included={ok} {reason}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["grok", "ring"])
    ap.add_argument("--conditions", nargs="+", default=["clean", "planted", "memorize"]); ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30000); ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    if a.kind == "grok":
        for cond in a.conditions:
            for s in (a.seeds or range(5)): train_grok(cond, s, a.epochs, max(1, a.epochs // 100))
    else:
        for s in (a.seeds or range(10)): train_ring(s, a.steps, max(1, a.steps // 50))
