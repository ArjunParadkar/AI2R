"""Ring-attractor RNN: angular integration.

The network sees a sequence of angular velocities and must report its heading.  The only way
to do that is to carry a heading around a circle, so the ground truth for P1 is exact: one
H^1 class whose coordinate is the heading.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn


class RingRNN(nn.Module):
    def __init__(self, n=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.W = nn.Parameter(0.9 * torch.eye(n) + torch.randn(n, n, generator=g) * (0.3 * n ** -0.5))   # near-identity: the cue must survive 40 steps
        self.U = nn.Parameter(torch.randn(1, n, generator=g) * 0.5)
        self.U0 = nn.Parameter(torch.randn(2, n, generator=g) * 0.5)   # initial-heading cue
        self.b = nn.Parameter(torch.zeros(n))
        self.V = nn.Parameter(torch.randn(n, 2, generator=g) * (n ** -0.5))

    def step(self, h, v):
        return torch.tanh(h @ self.W + v[:, None] * self.U + self.b)

    def readout(self, h):
        return h @ self.V

    def psi(self, h):
        """The remaining network from the hidden state to the output: one more recurrent step
        with zero velocity, then the readout.  Nonlinear, so the pullback metric varies."""
        return self.readout(torch.tanh(h @ self.W + self.b))

    def forward(self, vs, theta0=None):
        """theta0: initial heading, cued once at t=0 as (cos, sin).  Uniform theta0 is what
        forces a genuinely periodic code: without it the network learns an unwrapped line for
        heading over the few radians it ever sees, and there is no ring to find."""
        h = torch.zeros(vs.shape[0], self.n, device=vs.device)
        if theta0 is not None:
            h = torch.tanh(torch.stack([torch.cos(theta0), torch.sin(theta0)], 1) @ self.U0 + self.b)
        for t in range(vs.shape[1]):
            h = self.step(h, vs[:, t])
        return h


def batch(bs, T, device, rng, sigma=0.3):
    v = torch.as_tensor(rng.normal(0, sigma, size=(bs, T)), dtype=torch.float32, device=device)
    th0 = torch.as_tensor(rng.uniform(0, 2 * np.pi, size=bs), dtype=torch.float32, device=device)
    theta = th0 + torch.cumsum(v, 1)[:, -1]
    return v, torch.stack([torch.cos(theta), torch.sin(theta)], 1), theta, th0


def run(seed, steps, outdir, n=128, T=25, bs=256, lr=1e-3, device='cuda', ckpt_pct=5.0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    m = RingRNN(n, seed).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    every = max(1, int(steps * ckpt_pct / 100))
    hist, t0 = [], time.time()
    for s in range(steps + 1):
        Tt = int(rng.integers(10, 41))
        v, y, th, th0 = batch(bs, Tt, device, rng)
        h = m(v, th0); pred = m.readout(h)
        loss = ((pred - y) ** 2).mean()
        if s % every == 0 or s == steps:
            with torch.no_grad():
                vv, yy, tt, t0_ = batch(1024, T, device, np.random.default_rng(9999))
                hh = m(vv, t0_); pp = m.readout(hh)
                ang_err = torch.atan2(pp[:, 1], pp[:, 0]) - tt
                ang_err = torch.atan2(torch.sin(ang_err), torch.cos(ang_err)).abs().mean().item()
            hist.append(dict(step=s, loss=float(loss.item()), ang_err_rad=ang_err))
            torch.save(m.state_dict(), f'{outdir}/ckpt_{s:06d}.pt')
            if s % (every * 5) == 0:
                print(f'[ring s{seed}] {s:6d} loss {loss.item():.4f} ang_err {ang_err:.4f} rad', flush=True)
        if s == steps:
            break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    json.dump(dict(meta=dict(seed=seed, steps=steps, n=n, T=T, wall_s=time.time() - t0), history=hist),
              open(f'{outdir}/history.json', 'w'), indent=1)
    print(f'[ring s{seed}] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    run(a.seed, a.steps, a.outdir)
