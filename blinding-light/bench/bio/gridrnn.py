"""Path-integrating RNN (Sorscher/Xu-style; the lab's own configuration is not available on this machine, so this is our implementation) with an OPTIONAL
conformal-isometry loss.  Authors: Arjun Paradkar and Reva Paradkar.
  state   h_{t+1} = ReLU(W h_t + V v_t)            (N recurrent units, v_t = 2-D velocity)
  init    h_0 = ReLU(A p_0)                        (place-cell code of the start position)
  readout softmax(R h_t) over P place cells, target = Gaussian place code of the true position (cross-entropy)
  conformal isometry (optional): for random small displacements d (|d| = eps), rho = ||step(h, d) - h||^2 / eps^2 over states h seen on trajectories;
      L_conf = Var(rho) / mean(rho)^2   (scale-free: the pullback metric of the grid code is isotropic and position-independent)
  The conformal term acts DIRECTLY on the object the metric instrument reads: results for the conformal condition are partly by construction (registered)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

L_ARENA = 2.2


class GridRNN(nn.Module):
    def __init__(s, N=512, P=512, seed=0, L=L_ARENA):
        super().__init__(); torch.manual_seed(seed); s.N, s.P, s.L = N, P, L
        s.W = nn.Parameter(torch.randn(N, N) * (0.5 / N ** 0.5)); s.V = nn.Parameter(torch.randn(N, 2) * 0.5); s.A = nn.Linear(P, N, bias=False); s.R = nn.Linear(N, P, bias=False)
        g = torch.Generator().manual_seed(1234); s.register_buffer('centres', torch.rand(P, 2, generator=g) * L); s.sigma = 0.12

    def place(s, pos):                         # pos (..., 2) -> (..., P): softmax of a centre-surround (difference-of-Gaussians) code, temperature 0.1
        d2 = ((pos[..., None, :] - s.centres) ** 2).sum(-1); a = torch.exp(-d2 / (2 * s.sigma ** 2)) - 0.25 * torch.exp(-d2 / (8 * s.sigma ** 2)); return torch.softmax(a / 0.1, -1)

    def step(s, h, v): return F.relu(h @ s.W.T + v @ s.V.T)

    def run(s, p0, vel):                       # p0 (B,2), vel (B,T,2) -> states (B,T+1,N)
        h = F.relu(s.A(s.place(p0))); out = [h]
        for t in range(vel.shape[1]): h = s.step(h, vel[:, t]); out.append(h)
        return torch.stack(out, 1)


def trajectories(B, T, L=L_ARENA, speed=0.12, gen=None, device='cuda'):
    """random walk with reflecting walls; returns p0 (B,2), vel (B,T,2), pos (B,T+1,2).  speed = mean step length (m)."""
    p = torch.rand(B, 2, generator=gen, device=device) * L; pos = [p]; vel = []; ang = torch.rand(B, generator=gen, device=device) * 2 * np.pi
    for _ in range(T):
        ang = ang + 0.5 * torch.randn(B, generator=gen, device=device); sp = speed * (0.5 + torch.rand(B, generator=gen, device=device)); v = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * sp[:, None]
        q = p + v; refl = (q < 0) | (q > L); q = torch.where(q < 0, -q, q); q = torch.where(q > L, 2 * L - q, q); v = q - p; p = q; pos.append(p); vel.append(v)
    return pos[0], torch.stack(vel, 1), torch.stack(pos, 1)


def conformal_loss(m, states, eps=0.03, gen=None):
    h = states.reshape(-1, states.shape[-1])[torch.randint(0, states.shape[0] * states.shape[1], (1024,), device=states.device, generator=gen)]
    th = torch.rand(len(h), device=h.device, generator=gen) * 2 * np.pi; d = eps * torch.stack([torch.cos(th), torch.sin(th)], 1)
    rho = ((m.step(h, d) - m.step(h, torch.zeros_like(d))) ** 2).sum(1) / eps ** 2; return rho.var() / rho.mean().clamp_min(1e-8) ** 2


def train(cond, seed, steps=20000, B=128, T=40, lr=1e-3, wd=1e-4, lam_conf=1.0, ckpt_every=500, device='cuda', log=None):
    """cond: 'conformal' | 'noconf' | 'memorizer' | 'untrained'.  Returns model, history [(step, pi_loss, pi_err, conf)], checkpoints {step: state_dict}."""
    m = GridRNN(seed=seed).to(device); gen = torch.Generator(device=device).manual_seed(seed + 77); hist, ck = [], {}
    if cond == 'untrained': return m, hist, {0: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0.0)
    fixed = trajectories(64, T, gen=torch.Generator(device=device).manual_seed(seed + 5), device=device) if cond == 'memorizer' else None
    for step in range(steps + 1):
        if cond == 'memorizer': p0, vel, pos = fixed
        else: p0, vel, pos = trajectories(B, T, gen=gen, device=device)
        st = m.run(p0, vel); logits = m.R(st); tgt = m.place(pos); lp = F.log_softmax(logits, -1); loss_pi = -(tgt * lp).sum(-1).mean()
        loss = loss_pi + wd * (m.W ** 2).sum() + 0.0
        lc = torch.zeros((), device=device)
        if cond == 'conformal': lc = conformal_loss(m, st.detach() if False else st, gen=gen); loss = loss + lam_conf * lc
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % ckpt_every == 0:
            with torch.no_grad():
                p0, vel, pos = trajectories(256, T, gen=torch.Generator(device=device).manual_seed(999), device=device); st = m.run(p0, vel)
                dec = m.centres[m.R(st).argmax(-1)]; err = float((dec - pos).norm(dim=-1).mean())
            hist.append(dict(step=step, pi_loss=float(loss_pi.detach()), pi_err=err, conf=float(lc.detach()))); ck[step] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            if log and step % (ckpt_every * 4) == 0: log(f'{cond} s{seed} step {step} pi_loss {float(loss_pi):.3f} err {err:.3f} conf {float(lc):.3f}')
    return m, hist, ck


@torch.no_grad()
def ratemaps(m, bins=44, n_traj=1024, T=200, device='cuda', seed=0):
    """mean activity of every recurrent unit per spatial bin along long path-integrated trajectories (units that never fire are dropped by the caller)."""
    g = torch.Generator(device=device).manual_seed(seed + 31); L = m.L; acc = torch.zeros(bins, bins, m.N, device=device); cnt = torch.zeros(bins, bins, device=device)
    for _ in range(n_traj // 128):
        p0, vel, pos = trajectories(128, T, gen=g, device=device); st = m.run(p0, vel)
        ij = (pos / L * bins).long().clamp(0, bins - 1); flat = (ij[..., 0] * bins + ij[..., 1]).reshape(-1)
        acc.view(-1, m.N).index_add_(0, flat, st.reshape(-1, m.N)); cnt.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    rm = (acc / cnt.clamp_min(1)[..., None]).permute(2, 0, 1).cpu().numpy(); return rm, (cnt.cpu().numpy() > 0)
