"""Ring-attractor RNN: P1 (one circle, named by heading), P7 (its birth), P4 (off-ring states)."""
import sys, os, json, glob, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.ring_rnn import RingRNN, batch
from whiteboxtopo.extract import jacobians, logits_of
from whiteboxtopo.manifold import fit_manifold
from whiteboxtopo.topology import persistence, circular_coordinates, circular_correlation, circular_rank_correlation, isotypic_decomposition
from whiteboxtopo.geometry import reach, ignorance
from whiteboxtopo.nulls import feature_shuffle
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def cloud(m, n, seed, sigma=0.3, T=25):
    rng = np.random.default_rng(seed)
    v = torch.as_tensor(rng.normal(0, sigma, size=(n, T)), dtype=torch.float32, device='cuda')
    th0 = torch.as_tensor(rng.uniform(0, 2 * np.pi, size=n), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        h = m(v, th0)
    th = (th0 + torch.cumsum(v, 1)[:, -1]).cpu().numpy()
    return h.cpu().numpy().astype(np.float32), np.mod(th, 2 * np.pi)


def circle_of(X, theta, band=None):
    res = persistence(X)
    h1 = res['dgms'][1]
    if len(h1) == 0:
        return dict(n_sig=0, r=0.0, life=0.0, ratio=0.0)
    life = h1[:, 1] - h1[:, 0]
    srt = np.sort(life)
    j = int(np.argmax(life))
    cc = circular_coordinates(X, h1, res['cocycles'][1], D=res['D'], class_idx=j)
    r = circular_correlation(cc['theta'], theta) if cc else 0.0
    ratio = float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else float('inf')
    n_sig = int((life > band).sum()) if band is not None else int((life > 0.5 * srt[-1]).sum())
    return dict(n_sig=n_sig, r=float(r), life=float(srt[-1]), ratio=ratio,
                rel_life=float(srt[-1] / res['median_dist']))


NBIN = 72


def isotypic_circle(X, theta, seed=0):
    """Heading is the group coordinate.  Planes are fitted on half the cloud and the circle is
    read on the other half (out-of-sample), then certified by H^1 and the DSMVJ coordinate."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); a, b = perm[:len(X) // 2], perm[len(X) // 2:]
    bins = np.floor(theta / (2 * np.pi) * NBIN).astype(int) % NBIN
    dec, freqs = isotypic_decomposition(X[a], bins[a], NBIN, n_freq=2, detect='dft', var_floor=0.0, X_eval=X[b])
    best = dict(r=0.0, freq=None, ratio=0.0, var=0.0)
    for d in dec:
        Y = np.ascontiguousarray(d['proj']).astype(np.float32)
        res = persistence(Y); h1 = res['dgms'][1]
        if len(h1) == 0:
            continue
        life = h1[:, 1] - h1[:, 0]; j = int(np.argmax(life)); srt = np.sort(life)
        cc = circular_coordinates(Y, h1, res['cocycles'][1], D=res['D'], class_idx=j)
        if cc is None:
            continue
        r = circular_correlation(cc['theta'], d['freq'] * theta[b])
        if r > best['r']:
            best = dict(r=float(r), r_rank=float(circular_rank_correlation(cc['theta'], d['freq'] * theta[b])), freq=int(d['freq']), var=float(d['var']),
                        ratio=float(srt[-1] / max(srt[-2], 1e-9)) if len(srt) > 1 else 99.0,
                        n_long=int((life > 0.5 * srt[-1]).sum()))
    return best


def main():
    out = {}
    for seed in (0, 1, 2):
        run = f'runs/ring_s{seed}'
        cks = sorted(glob.glob(f'{run}/ckpt_*.pt'))
        hist = json.load(open(f'{run}/history.json'))['history']
        rec = dict(sweep=[])
        for ck, h in zip(cks, hist):
            m = RingRNN(seed=seed).cuda(); m.load_state_dict(torch.load(ck)); m.eval()
            X, th = cloud(m, 600, seed=123)
            c = circle_of(X, th)
            ic = isotypic_circle(X, th)
            rec['sweep'].append(dict(step=h['step'], ang_err=h['ang_err_rad'], iso_r=ic['r'], iso_ratio=ic['ratio'], iso_freq=ic['freq'], **c))
        # final checkpoint, full treatment
        t0 = time.time()
        X, th = cloud(m, 1500, seed=7)
        J = jacobians(m.psi, X).astype(np.float32)
        fit = fit_manifold(X, J, metric='g')
        Xs = feature_shuffle(X, 0)
        rn = persistence(Xs, do_cocycles=False)
        ln = rn['dgms'][1][:, 1] - rn['dgms'][1][:, 0]
        band = float(ln.max()) if len(ln) else 0.0
        c = circle_of(X, th, band=band)
        cn = circle_of(Xs, th)
        ic = isotypic_circle(X, th); icn = isotypic_circle(Xs, th)
        rec['final'] = dict(m=fit.m, d=float(np.nanmedian(fit.d_local)), noise_band=band,
                            n_significant_H1=c['n_sig'], r_heading=c['r'], lifetime=c['life'],
                            life_ratio=c['ratio'], null_r=cn['r'], null_top_life=cn['life'],
                            iso=ic, iso_null=icn,
                            wall_s=time.time() - t0)
        # birth: first checkpoint from which r>0.9 holds to the end
        ok = [s['iso_r'] > 0.9 and s['iso_ratio'] > 3 and s['iso_freq'] == 1 for s in rec['sweep']]
        birth = next((i for i in range(len(ok)) if all(ok[i:])), None)
        errs = np.array([s['ang_err'] for s in rec['sweep']])
        drop = int(np.argmax(errs < 0.3)) if (errs < 0.3).any() else None     # integration works: < 0.3 rad
        rec['P7'] = dict(birth_ckpt=birth, birth_step=rec['sweep'][birth]['step'] if birth is not None else None,
                         err_drop_ckpt=drop, err_drop_step=rec['sweep'][drop]['step'] if drop is not None else None)
        # P4: off-ring states
        Xh, _ = cloud(m, 800, seed=99)
        eps, _, nnb = reach(fit.X, fit.bases, fit.J, Xh, 99)
        _, dg_in = ignorance(Xh, fit.X, fit.bases, fit.J, eps, nnb)
        nn_e = NearestNeighbors(n_neighbors=1).fit(fit.X)
        e_in = nn_e.kneighbors(Xh)[0][:, 0]
        conf = lambda A: -np.abs(np.linalg.norm(logits_of(m.psi, A), axis=1) - 1.0)   # output-norm confidence
        c_in = conf(Xh)
        rng = np.random.default_rng(5)
        sets = {'radial_shrink': Xh * 0.6,
                'gaussian_state': (rng.standard_normal(Xh.shape) * X.std()).astype(np.float32),
                'big_velocity': cloud(m, 800, seed=42, sigma=3.0)[0]}
        p4 = {}
        for nm, Xo in sets.items():
            _, dg_o = ignorance(Xo, fit.X, fit.bases, fit.J, eps, nnb)
            y = np.r_[np.zeros(len(dg_in)), np.ones(len(dg_o))]
            p4[nm] = dict(auc_ignorance=float(roc_auc_score(y, np.r_[dg_in, dg_o])),
                          auc_output_conf=float(roc_auc_score(y, np.r_[-c_in, -conf(Xo)])),
                          auc_euclid_knn=float(roc_auc_score(y, np.r_[e_in, nn_e.kneighbors(Xo)[0][:, 0]])))
        rec['P4'] = dict(reach=float(eps), sets=p4)
        out[f's{seed}'] = rec
        print(f"[ring s{seed}] m={fit.m:.2f} H1_sig={c['n_sig']} r(heading)={c['r']:.4f} "
              f"ratio={c['ratio']:.1f} null_r={cn['r']:.3f} | ISO r={ic['r']:.4f} r_rank={ic.get('r_rank',0):.4f} k={ic['freq']} ratio={ic['ratio']:.1f} n_long={ic.get('n_long')} null={icn['r']:.3f} | birth ckpt {birth} err-drop ckpt {drop}", flush=True)
        for nm, v in p4.items():
            print(f"     P4 {nm:15s} ignorance {v['auc_ignorance']:.3f} | conf {v['auc_output_conf']:.3f} | euclid {v['auc_euclid_knn']:.3f}", flush=True)
    os.makedirs('results', exist_ok=True)
    json.dump(out, open('results/ring.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
