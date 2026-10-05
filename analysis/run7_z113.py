"""Run 7, Z113: X.4 check (lambda for swap / orbit translation / symmetric translation), P5' (natural failure on early-stopped seeds), R (reconstruction).
usage: run7_z113.py X | P5 | R.  CPU.  Cited: LieGG; Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018."""
import sys, os, json, glob, re, time
os.environ.setdefault('WB_DEV', 'cpu')
import numpy as np, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.z113 import build_data, P, NULL, EQ
from analysis.run4_z113 import load, group
from analysis.run5_z113 import finite_tower
from analysis.run6_z113 import softmax_scores
from whiteboxtopo.symmetry_finite import cycle_coordinate, characters
from whiteboxtopo.knowledge import orbit_stats, z113_group
R = 'results/run7'; os.makedirs(R, exist_ok=True)
CLEAN = [(f'r{5 if s < 12 else 6}_z113_clean_s{s}', s) for s in (9, 10, 11, 12, 13, 14)]; MEM = [(f'r6_z113_scrambled_decoy_s{s}', s) for s in (15, 16, 17)]
LAY = ('embed', 'resid_mid', 'resid_post', 'logits')


def block_X():
    out = {}
    for run, s in CLEAN + MEM:
        cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt')); m = load(cks[-1], s); m0 = load(cks[0], s); _, _, sig = group(m); u = int(sig[0]) if sig is not None else 1; t0 = time.time()
        qt, qu = finite_tower(m, u, n=400), finite_tower(m0, u, n=400); lam = {T: {L: (qu[T][L] - qt[T][L]) / max(qu[T][L], 1e-9) for L in LAY} for T in ('swap', 'translation', 'orbit_translation', 'hint_change')}
        out[run] = dict(seed=s, u=u, q_trained=qt, q_untrained=qu, lambda_=lam)
        print(f"[X-Z] {run}: λ_post swap {lam['swap']['resid_post']:+.2f} orbit-transl {lam['orbit_translation']['resid_post']:+.2f} sym-transl {lam['translation']['resid_post']:+.2f} hint {lam['hint_change']['resid_post']:+.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/X_z113.json', 'w'), indent=1, default=float)


def block_P5():
    out = {}
    for s in (18, 19, 20):
        run = f'r7_z113_clean_s{s}'; H = json.load(open(f'runs/{run}/history.json'))['history']; hist = {h['step']: h for h in H}; cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt'))
        steps = [int(re.search(r'ckpt_(\d+)', c)[1]) for c in cks]; te = [hist.get(st, {}).get('test_acc', 0) for st in steps]
        win = [i for i, a in enumerate(te) if 0.70 <= a <= 0.90]; i = win[0] if win else int(np.argmin([abs(a - 0.8) for a in te])); m = load(cks[i], s); t0 = time.time()
        _, r, sig = group(m); u = int(sig[0]) if sig is not None else None
        d = build_data(s, 'clean', frac=0.3); tk = d['test_toks'][:2000].copy(); tk[:, 2] = NULL; y = d['test_y'][:2000]
        with torch.no_grad(): pred = m(torch.as_tensor(tk)).argmax(1).numpy()
        err = pred != y; G = z113_group(u if u else 1); S = orbit_stats(lambda t: m(t), torch.as_tensor(tk), G); sm = softmax_scores(m, tk); f = lambda v: float(roc_auc_score(err, -v))
        out[run] = dict(seed=s, ckpt=i, step=steps[i], test_acc=te[i], generator_found=bool(sig is not None), u=u, error_rate=float(err.mean()), auroc_kappa=f(S['kappa']), auroc_kappa_s=f(S['kappa_s']), auroc_msp=f(sm['msp']), auroc_entropy=f(sm['neg_entropy']),
                        kappa_mean_correct=float(S['kappa'][~err].mean()), kappa_mean_error=float(S['kappa'][err].mean()), wall_s=time.time() - t0)
        # null: kappa under 7 random token substitutions (rho = id)
        rng = np.random.default_rng(0); Gr = []
        for k in range(7):
            sl, tok = int(rng.integers(0, 2)), int(rng.integers(1, P))
            def T(x, sl=sl, tok=tok):
                z = x.clone(); z[:, sl] = (z[:, sl] + tok) % P; return z
            Gr.append((T, lambda l: l))
        out[run]['auroc_kappa_random_transforms'] = f(orbit_stats(lambda t: m(t), torch.as_tensor(tk), Gr)['kappa'])
        print(f"[P5'-Z] {run}: ckpt {i} test {te[i]:.3f} gen {u} | err {err.mean():.3f} | AUROC κ {out[run]['auroc_kappa']:.3f} κs {out[run]['auroc_kappa_s']:.3f} msp {out[run]['auroc_msp']:.3f} ent {out[run]['auroc_entropy']:.3f} random {out[run]['auroc_kappa_random_transforms']:.3f} | κ correct {out[run]['kappa_mean_correct']:.2f} error {out[run]['kappa_mean_error']:.2f}", flush=True)
        json.dump(out, open(f'{R}/P5_z113.json', 'w'), indent=1, default=float)


def fourier_features(a, b, c, ks, t):
    ph = 2 * np.pi * ((t[a] + t[b] - t[c]) % P) / P; return np.stack([np.cos(k * ph) for k in ks] + [np.sin(k * ph) for k in ks] + [np.ones_like(ph)], -1)


def block_R():
    out = {}
    for run, s in CLEAN:
        cks = sorted(glob.glob(f'runs/{run}/ckpt_*.pt')); m = load(cks[-1], s); _, _, sig = group(m)
        if sig is None:
            out[run] = dict(note='no generator'); continue
        t = cycle_coordinate(sig); ch = characters(m.W_E[:P].numpy(), t, top=5); ks = [c['freq'] for c in ch]; t0 = time.time()
        d = build_data(s, 'clean', frac=0.3); tr = d['train_toks'][np.random.default_rng(0).choice(len(d['train_toks']), 1000, replace=False)]; te = d['test_toks'][:2000]
        cc = np.arange(P); A = tr[:, 0][:, None].repeat(P, 1); B = tr[:, 1][:, None].repeat(P, 1); Cc = cc[None].repeat(len(tr), 0)
        Phi = fourier_features(A.ravel(), B.ravel(), Cc.ravel(), ks, t).reshape(len(tr), P, -1); Y = ((tr[:, 0] + tr[:, 1]) % P)[:, None] == cc[None]
        w, *_ = np.linalg.lstsq(Phi.reshape(-1, Phi.shape[-1]), Y.ravel().astype(float), rcond=None)
        def rebuilt_logits(tk, w=w, ks=ks):
            A = tk[:, 0][:, None].repeat(P, 1); B = tk[:, 1][:, None].repeat(P, 1); Cc = cc[None].repeat(len(tk), 0); return (fourier_features(A.ravel(), B.ravel(), Cc.ravel(), ks, t).reshape(len(tk), P, -1) @ w)
        acc_re = float((rebuilt_logits(te).argmax(1) == (te[:, 0] + te[:, 1]) % P).mean())
        tk2 = te.copy(); tk2[:, 2] = NULL
        with torch.no_grad(): L = m(torch.as_tensor(tk2)).float().numpy()
        acc_orig = float((L.argmax(1) == (te[:, 0] + te[:, 1]) % P).mean())
        # ablation symmetry: drop frequency k in the rebuild (zero its two weights) / project the k-th Fourier mode out of W_E in the original
        nf = len(ks); abl = []
        for j, k in enumerate(ks):
            w2 = w.copy(); w2[j] = 0; w2[nf + j] = 0; a_re = float((rebuilt_logits(te, w2).argmax(1) == (te[:, 0] + te[:, 1]) % P).mean())
            m2 = load(cks[-1], s); E = m2.W_E.data[:P].clone(); order = np.argsort(t); Ec = E[order]; Fk = torch.fft.rfft(Ec, dim=0); Fk[k] = 0; Ec2 = torch.fft.irfft(Fk, n=P, dim=0); E2 = torch.empty_like(E); E2[order] = Ec2; m2.W_E.data[:P] = E2
            with torch.no_grad(): a_or = float((m2(torch.as_tensor(tk2)).argmax(1).numpy() == (te[:, 0] + te[:, 1]) % P).mean())
            abl.append(dict(k=int(k), rebuilt_drop=acc_re - a_re, original_drop=acc_orig - a_or, same_direction=bool((acc_re - a_re > 0.005) == (acc_orig - a_or > 0.005))))
        # coverage gap gamma: residual of (affinely matched) original vs rebuilt logits; structured fraction above the MP edge of shuffled residuals
        Lr = rebuilt_logits(te); Lr = (Lr - Lr.mean(1, keepdims=True)) / Lr.std(1, keepdims=True).clip(1e-9); Lo = (L - L.mean(1, keepdims=True)) / L.std(1, keepdims=True).clip(1e-9)
        alpha = float((Lo * Lr).sum() / (Lr * Lr).sum()); res = Lo - alpha * Lr; ev = np.linalg.eigvalsh(np.cov(res.T)); rng = np.random.default_rng(0)
        edge = max(np.linalg.eigvalsh(np.cov(np.stack([rng.permutation(col) for col in res.T]).T)).max() for _ in range(5)); gamma = float(ev[ev > edge].sum() / ev.sum()); expl = float(1 - (res ** 2).sum() / (Lo ** 2).sum())
        out[run] = dict(seed=s, u=int(sig[0]), frequencies_cycle_coord=ks, rebuilt_acc=acc_re, original_acc=acc_orig, ablation=abl, n_same_direction=int(sum(a['same_direction'] for a in abl)), gamma=gamma, logit_variance_explained=expl, wall_s=time.time() - t0)
        print(f"[R] {run}: K {ks} | rebuilt {acc_re:.4f} (orig {acc_orig:.4f}) | ablation same-direction {out[run]['n_same_direction']}/5 " + ' '.join(f"k{a['k']}:{a['rebuilt_drop']:+.2f}/{a['original_drop']:+.2f}" for a in abl) + f" | γ {gamma:.3f} (logit var explained {expl:.2f}) ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f'{R}/R.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    {'X': block_X, 'P5': block_P5, 'R': block_R}[sys.argv[1]]()
