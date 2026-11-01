"""P3: detect a spurious class during training and interrupt it.

Detection (spec 3.5).  A persistent H^1 class c is spurious at time t when
  (i)   its circular coordinate tracks a nuisance variable z  (|corr| > 0.5),
  (ii)  the output actually depends on movement along it,  E||J v_c||^2 > tau,
  (iii) on a probe where z is decorrelated from y, z carries no information about y.
Here z is the planted hint token.  (iii) holds by construction: the decorrelated probe draws
the hint uniformly, independent of the answer.

Interrupt.  While the class survives, add  L_c = lambda * E||J_l(h) v_c(h)||^2  to the loss --
output invariance along the spurious coordinate.  v_c is the direction the representation moves
when the hint is advanced by one residue, which is the tangent to the coordinate's level
structure; the directional derivative is taken with a forward-mode JVP, so the penalty costs
one extra pass rather than a full Jacobian.
"""
import sys, os, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import Z113Transformer, build_data, P, NULL, EQ
from systems.z113_probe import psi_of
from whiteboxtopo.topology import isotypic_decomposition, persistence, circular_coordinates, circular_correlation
from whiteboxtopo.extract import jacobians


def resid_mid(model, toks):
    x = model.embed(toks)
    return (x + model.attn(x))[:, -1, :]


def hint_probe(d, n=113, seed=0, device='cuda'):
    """Fix a block of (a,b) pairs and sweep the hint token over all of Z113.
    The group parameter is the hint, so the isotypic machinery reads off exactly the
    representation of the nuisance."""
    rng = np.random.default_rng(seed)
    tr = d['train_toks']
    base = tr[rng.choice(len(tr), size=n, replace=False)]
    h = np.arange(P)
    toks = np.stack([base[:, 0], base[:, 1], h, np.full(n, EQ)], 1)
    return torch.as_tensor(toks, device=device), h, base


def spurious_report(model, d, tau=None, device='cuda', r_thresh=0.5):
    """Is there a persistent class coordinatised by the hint that the output depends on?"""
    toks, h, base = hint_probe(d, device=device)
    with torch.no_grad():
        X = resid_mid(model, toks).cpu().numpy().astype(np.float32)
    dec, freqs = isotypic_decomposition(X, h, P, n_freq=4, detect='dft', var_floor=0.0)
    psi = psi_of(model, 'resid_mid')
    out = []
    for dd in dec:
        k = dd['freq']
        Y = np.ascontiguousarray(dd['proj']).astype(np.float32)
        res = persistence(Y)
        h1 = res['dgms'][1]
        if len(h1) == 0:
            continue
        life = h1[:, 1] - h1[:, 0]
        j = int(np.argmax(life))
        srt = np.sort(life)
        ratio = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else np.inf
        cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)
        if cc is None:
            continue
        r = circular_correlation(cc['theta'], 2 * np.pi * k * h / P)
        # (ii) does the output move when we move along this coordinate?
        v = hint_tangent(model, toks)
        sens = output_sensitivity(psi, resid_mid(model, toks), v)
        out.append(dict(freq=int(k), var=float(dd['var']), corr_z=float(r), life_ratio=ratio,
                        _geom=(dd['basis'].astype(np.float32), dd['dual'].astype(np.float32), dd['mean'].astype(np.float32)),
                        sensitivity=float(sens),
                        spurious=bool(r > r_thresh and ratio > 3.0 and dd['var'] > 0.02
                                      and (tau is None or sens > tau))))
    return out


def hint_tangent(model, toks):
    """v_c: the unit direction the representation moves when the hint advances one residue."""
    with torch.no_grad():
        h0 = resid_mid(model, toks)
        t2 = toks.clone()
        t2[:, 2] = (t2[:, 2] + 1) % P
        v = resid_mid(model, t2) - h0
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)


def output_sensitivity(psi, h, v, eps=1e-2):
    """E ||J(h) v||^2, by a central difference along v."""
    with torch.no_grad():
        d = (psi(h + eps * v) - psi(h - eps * v)) / (2 * eps)
    return float((d ** 2).sum(1).mean().item())


def invariance_penalty(model, toks=None, n=4096):
    """Output invariance along the spurious coordinate:

        L_c = E_{(a,b), h}  KL( p(. | a,b,h) || p(. | a,b,h+1) )   (symmetrised)

    This is the spec's lambda * E||J_l(h) v_c(h)||^2 -- the change in output for one step along
    the hint coordinate -- with three corrections that training forced on it, each of which the
    network exploited when it was missing:

      * the JVP form carried no gradient to the parameters (torch.func.jvp, closure weights);
      * E_h must run over the WHOLE circle and over ALL inputs, not the hint values and (a,b)
        pairs that occur in training: invariance enforced only there is simply memorised
        there, and the copy circuit survives everywhere else.  The penalty needs no labels, so
        it is evaluated on uniformly random (a,b,h);
      * measured in logit space and normalised, the penalty is driven to zero by inflating the
        logits.  Measured between output distributions there is nothing to inflate.
    """
    dev = model.W_E.device
    ab = torch.randint(0, P, (n, 2), device=dev)
    h = torch.randint(0, P, (n, 1), device=dev)
    eq = torch.full((n, 1), EQ, device=dev)
    lp0 = F.log_softmax(model(torch.cat([ab, h, eq], 1)), -1)
    lp1 = F.log_softmax(model(torch.cat([ab, (h + 1) % P, eq], 1)), -1)
    # the nuisance z is the hint's PRESENCE as well as its value (spec: "z = presence of the
    # token"); value-invariance alone leaves "a hint is present => the answer is a leaked
    # residue" intact, and decorrelated accuracy stalls at 13%.
    lpn = F.log_softmax(model(torch.cat([ab, torch.full_like(h, NULL), eq], 1)), -1)
    skl = lambda p, q: 0.5 * ((p.exp() * (p - q)).sum(1) + (q.exp() * (q - p)).sum(1)).mean()
    return skl(lp0, lp1) + skl(lp0, lpn)


def structure_penalty(model, toks, geoms):
    """The spec's L_c, taken literally: output invariance ALONG THE DISCOVERED COORDINATE, at the
    layer where the class lives.  Nothing here knows which input token the nuisance is.

    For each spurious class we hold its isotypic plane (basis e1,e2 and dual coordinates z).  The
    group moves a point h = mean + z1 e1 + z2 e2 along d/dphi = -z2 e1 + z1 e2, so one step of the
    discovered coordinate (one residue, angle 2 pi k / P) displaces h by
        delta(h) = (2 pi k / P) (-z2 e1 + z1 e2).
    The penalty is the symmetrised KL between the output at h and at h + delta(h), over the
    training batch.  delta is detached (it is the direction field); gradients reach the MLP and
    unembedding through psi and the embedding/attention through h.
    """
    h = resid_mid(model, toks)
    psi = lambda x: (x + model.mlp(x)) @ model.W_U
    lp0 = F.log_softmax(psi(h), -1)
    pen = 0.0
    for k, E, Dl, mu in geoms:
        with torch.no_grad():
            z = (h - mu) @ Dl
            delta = (2 * np.pi * k / P) * (-z[:, 1:2] * E[:, 0] + z[:, 0:1] * E[:, 1])
        lp1 = F.log_softmax(psi(h + delta), -1)
        pen = pen + 0.5 * ((lp0.exp() * (lp0 - lp1)).sum(1) + (lp1.exp() * (lp1 - lp0)).sum(1)).mean()
    return pen


def run(seed, steps, outdir, mode='interrupt', lam0=3.0, K=400, max_rounds=3,
        check_every=500, device='cuda', frac=0.5, tau=None, esc=10.0, save_ckpt=False):
    os.makedirs(outdir, exist_ok=True)
    d = build_data(seed, 'shortcut', frac=frac)
    T = lambda x: torch.as_tensor(x, device=device)
    xtr, ytr = T(d['train_toks']), T(d['train_y'])
    xte, yte = T(d['test_toks']), T(d['test_y'])
    xdec, xabl = T(d['dec_toks']), T(d['abl_toks'])
    worst = T(d['dec_worst'])
    leaky = T(d['train_leaky'])

    torch.manual_seed(seed)
    model = Z113Transformer(seed=seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    psi = psi_of(model, 'resid_mid')
    probe_toks, hh, _ = hint_probe(d, device=device)

    hist, events = [], []
    lam, rounds, penalty_on = 0.0, 0, False
    weights = None            # SPARE-style importance weights (mode == 'spare')
    geoms = []                # planes of the detected classes (mode == 'structure')
    t0 = time.time()
    for step in range(steps + 1):
        if step % check_every == 0:
            model.eval()
            with torch.no_grad():
                ltr, lte = model(xtr), model(xte)
                ld_, la_ = model(xdec), model(xabl)
                rec = dict(step=step,
                           train_acc=(ltr.argmax(-1) == ytr).float().mean().item(),
                           test_acc=(lte.argmax(-1) == yte).float().mean().item(),
                           dec_acc=(ld_.argmax(-1) == yte).float().mean().item(),
                           abl_acc=(la_.argmax(-1) == yte).float().mean().item(),
                           dec_worst_acc=(ld_.argmax(-1) == yte)[worst].float().mean().item(),
                           lam=lam)
                al = model(torch.cat([xtr[:, :2], torch.full_like(xtr[:, 2:3], NULL), xtr[:, 3:]], 1))
                rec['leaky_acc'] = (ltr.argmax(-1) == ytr)[leaky].float().mean().item()
                rec['leaky_acc_ablated'] = (al.argmax(-1) == ytr)[leaky].float().mean().item()
                rec['shortcut_reliance'] = rec['leaky_acc'] - rec['leaky_acc_ablated']
            sp = spurious_report(model, d, tau=tau)
            flagged = [c for c in sp if c['spurious']]
            if mode == 'structure' and flagged:     # refresh the direction field from the current planes
                geoms = [(c['freq'],) + tuple(torch.as_tensor(g, device=device) for g in c['_geom']) for c in flagged]
            for c in sp:
                c.pop('_geom', None)
            rec['spurious'] = sp
            rec['n_spurious'] = int(sum(s['spurious'] for s in sp))
            hist.append(rec)
            if save_ckpt:
                torch.save(model.state_dict(), f'{outdir}/ckpt_{step:06d}.pt')
            json.dump(dict(meta=dict(seed=seed, mode=mode, steps=steps, lam0=lam0, esc=esc, check_every=check_every, partial=True), history=hist, events=events),
                      open(f'{outdir}/history_partial.json', 'w'), indent=1, default=float)
            print(f"[{mode} s{seed}] {step:6d} test {rec['test_acc']:.3f} dec {rec['dec_acc']:.3f} "
                  f"worst {rec['dec_worst_acc']:.3f} reliance {rec['shortcut_reliance']:.3f} "
                  f"spurious {rec['n_spurious']} lam {lam:.1f}", flush=True)
            if mode in ('interrupt', 'structure') and rec['n_spurious'] > 0 and rounds < max_rounds:
                # newly detected, or it survived the previous round: escalate
                lam = lam0 if lam == 0 else lam * esc
                rounds += 1
                penalty_on = True
                events.append(dict(step=step, lam=float(lam), round=rounds,
                                   detected=[s for s in sp if s['spurious']],
                                   reliance=rec['shortcut_reliance']))
                print(f"   -> INTERRUPT round {rounds}: lambda={lam:g}", flush=True)
            elif mode in ('interrupt', 'structure') and penalty_on and rec['n_spurious'] == 0:
                print(f"   -> class gone; penalty stays on at lambda={lam:g}", flush=True)
            model.train()
        if step == steps:
            break
        if mode == 'spare' and step == check_every and weights is None:
            # SPARE (Yang et al. 2024), reduced: cluster each class's early representations,
            # then sample inversely to cluster size so minority groups are up-weighted.
            from sklearn.cluster import KMeans
            with torch.no_grad():
                Z = resid_mid(model, xtr).cpu().numpy()
            w = np.ones(len(Z)); yn = ytr.cpu().numpy()
            for c in np.unique(yn):
                ii = np.where(yn == c)[0]
                if len(ii) < 6:
                    continue
                lab = KMeans(2, n_init=3, random_state=0).fit_predict(Z[ii])
                for g in (0, 1):
                    gi = ii[lab == g]
                    if len(gi):
                        w[gi] = len(ii) / (2.0 * len(gi))
            weights = torch.as_tensor(w / w.mean(), device=device, dtype=torch.float32)
            events.append(dict(step=step, spare_weight_max=float(w.max()), spare_weight_min=float(w.min())))
        opt.zero_grad(set_to_none=True)
        if weights is not None:
            loss = (F.cross_entropy(model(xtr), ytr, reduction='none') * weights).mean()
        else:
            loss = F.cross_entropy(model(xtr), ytr)
        if lam > 0 and penalty_on and mode == 'structure':
            loss = loss + lam * structure_penalty(model, xtr, geoms)
        elif lam > 0 and penalty_on:
            loss = loss + lam * invariance_penalty(model, xtr)   # over the whole training distribution
        loss.backward()
        opt.step()
    json.dump(dict(meta=dict(seed=seed, mode=mode, steps=steps, lam0=lam0, K=K, tau=tau,
                             wall_s=time.time() - t0), history=hist, events=events),
              open(f'{outdir}/history.json', 'w'), indent=1)
    print(f'[{mode} s{seed}] done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=25000)
    ap.add_argument('--mode', default='interrupt'); ap.add_argument('--outdir', required=True)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--esc', type=float, default=10.0); ap.add_argument('--check-every', type=int, default=500)
    ap.add_argument('--max-rounds', type=int, default=3); ap.add_argument('--save-ckpt', action='store_true')
    a = ap.parse_args(); run(a.seed, a.steps, a.outdir, mode=a.mode, tau=a.tau, esc=a.esc, check_every=a.check_every, max_rounds=a.max_rounds, save_ckpt=a.save_ckpt)
