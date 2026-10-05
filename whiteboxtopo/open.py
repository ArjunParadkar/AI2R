"""open(model) -- the seven answers, in one call.

    report = open_model(layers={'resid_mid': layer_data}, holdout=..., ood=..., ...)
    print(report.summary())

Every answer carries the manifold score of the layer it came from.  When that score is low the
answer is flagged, because the definitions describe a manifold and a low score says the cloud
is not behaving like one -- that is the honesty mechanism, and it is not optional.
"""
import json, time
import numpy as np

from .manifold import fit_manifold
from .problems import (p1_symmetry_inventory, p1_inventory, p2_explain, p4_ignorance, p5_failure,
                       p6_geodesic_generalization, p3_spurious, fiber_beta0)
from .nulls import feature_shuffle
from .topology import persistence

LOW_M = 0.5


class Report(dict):
    """Typed answers plus timings.  dict so it serialises; attribute access for convenience."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def summary(self):
        L = []
        A = self.get('answers', {})
        for layer, ans in A.items():
            m = ans.get('m', float('nan'))
            flag = '  [LOW MANIFOLD SCORE -- answers below are not trustworthy]' if m < LOW_M else ''
            L.append(f"layer {layer}   manifold score m = {m:.2f}{flag}")
            if 'P1' in ans:
                p = ans['P1']
                names = [c.get('label', 'unnamed') for c in p.get('coord_summary', [])]
                L.append(f"  P1  what it learned: beta0={p['beta0']}, {p['beta1']} persistent "
                         f"H1 class(es); coordinates: {', '.join(names) if names else 'none named'}")
                if 'P1_symmetry' in ans:
                    cs = [c for c in ans['P1_symmetry'] if c['present']]
                    L.append("      with the task symmetry supplied: " + (", ".join(
                        f"circle at frequency {c['freq']} (r={c['r']:.3f}, {100*c['var']:.0f}% var)" for c in cs) or 'no certified circle'))
                if p.get('fiber_summary'):
                    L.append(f"      fibers: {p['fiber_summary']}")
            if 'P6' in ans:
                p = ans['P6']
                L.append(f"  P6  understanding vs memorization: G = {p['G']:.3f} "
                         f"(interior-only {p.get('G_interior', float('nan')):.3f}); "
                         f"mean beta0 of a fiber = {p['beta0_fiber_mean']:.2f} "
                         f"({'geodesically closed -- understood' if p['G'] > 0.8 and p['beta0_fiber_mean'] < 1.5 else 'not closed -- memorizing'})")
            if 'P2' in ans and ans['P2']:
                e = ans['P2'][0]
                L.append(f"  P2  why this decision (input {e['i']}): predicted {e['pred']}, nearest "
                         f"other answer {e['flips_to']} at manifold margin {e['margin']:.3f} "
                         f"({e['n_steps']} steps)")
            if 'P4' in ans:
                p = ans['P4']
                bits = ', '.join(f"{k}: AUC {v['auc_ignorance']:.3f} (softmax {v.get('auc_softmax', float('nan')):.3f}, "
                                 f"euclid {v['auc_euclid_knn']:.3f}; tubular [post-hoc] {v.get('auc_tubular', float('nan')):.3f})" for k, v in p['sets'].items())
                L.append(f"  P4  what it does not know: reach = {p['reach']:.3f}; {bits}")
            if 'P5' in ans:
                p = ans['P5']
                L.append(f"  P5  where it will fail: Spearman(rho, failure radius) = "
                         f"{p['spearman_rho']:.3f} (p={p['p_rho']:.1e}), softmax margin "
                         f"{p['spearman_softmax_margin']:.3f}; {len(p['saddles'])} confidence saddles")
            if 'P3' in ans:
                p = ans['P3']
                L.append(f"  P3  when bias entered: {p['n_spurious']} spurious class(es)"
                         + (f", born at step {p['birth_step']}" if p.get('birth_step') is not None else ''))
            if 'P7' in ans:
                p = ans['P7']
                L.append(f"  P7  how it evolved: {p['n_surviving']} surviving structure(s); "
                         f"birth order {p['birth_order']}")
        L.append(f"\ntotal wall time {self.get('wall_s', 0):.1f}s")
        return '\n'.join(L)

    def to_json(self, path=None):
        def enc(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            return str(o)
        s = json.dumps(self, default=enc, indent=1)
        if path:
            open(path, 'w').write(s)
        return s


def open_model(layers, holdout=None, ood=None, failure_radius=None, nuisance=None,
               explain_inputs=(), psi=None, metric='g', k=15, noise_band='null',
               do_fibers=True, verbose=False, p1_metric='euclidean', symmetry=None):
    """Run every problem that the supplied data makes answerable, on every layer."""
    t00 = time.time()
    rep = Report(answers={}, timings={}, config=dict(metric=metric, k=k))
    for name, ld in layers.items():
        t = {}
        t0 = time.time(); fit = fit_manifold(ld.X, ld.J, k=k, metric=metric); t['fit'] = time.time() - t0
        ans = dict(m=fit.m, d_intrinsic=float(np.nanmedian(fit.d_local)), beta0=int(fit.n_components()))

        band = None
        if noise_band == 'null':
            t0 = time.time()
            # The band is 1.5x the longest H^1 bar over two independent shuffles.  The longest bar
            # of ONE shuffle is exchangeable with the longest bar of a structureless cloud, so
            # using it bare flags pure noise half the time (measured: 6 of 12 Gaussian clouds).
            # Noise stays under 1.5x; a real circle clears 4x.
            tops = []
            for sd in (0, 1):
                rn = persistence(feature_shuffle(ld.X, seed=sd), maxdim=1, do_cocycles=False)
                lf = rn['dgms'][1][:, 1] - rn['dgms'][1][:, 0]
                lf = lf[np.isfinite(lf)]
                tops.append(float(lf.max()) if len(lf) else 0.0)
            band = 1.5 * max(tops)
            ans['noise_band_from_null'] = band
            t['null'] = time.time() - t0

        t0 = time.time()
        p1 = p1_inventory(ld, fit, noise_band=(band if p1_metric == 'euclidean' else None),
                          do_fibers=do_fibers, metric=p1_metric)
        p1['coord_summary'] = [dict(label=(c['naming']['named']['name'] +
                                           (f"@freq{c['naming']['named']['freq']}" if c['naming']['named'].get('freq') else '')
                                           if c['naming']['named'] else 'unnamed'),
                                    lifetime=c['lifetime']) for c in p1['coords']]
        if do_fibers and 'fibers' in p1:
            b1 = [f['beta1'] for f in p1['fibers'].values() if not f.get('skipped')]
            p1['fiber_summary'] = (f"{len(b1)} fibers, mean beta1 = {np.mean(b1):.2f}" if b1 else 'none')
        ans['P1'] = {k2: v for k2, v in p1.items() if k2 not in ('dgm1', 'coords', 'fibers')}
        ans['_p1_full'] = p1
        t['P1'] = time.time() - t0

        if symmetry and name in symmetry:
            # symmetry = {layer: dict(X=orbit activations, param=group parameter, period=N)}
            t0 = time.time(); sy = symmetry[name]
            ans['P1_symmetry'] = p1_symmetry_inventory(sy['X'], sy['param'], sy['period'], n_freq=sy.get('n_freq', 6),
                                                       X_eval=sy.get('X_eval'), param_eval=sy.get('param_eval'))
            t['P1_symmetry'] = time.time() - t0

        t0 = time.time()
        g = p6_geodesic_generalization(ld, fit)
        b0 = fiber_beta0(ld, fit)
        g['beta0_fiber_mean'] = float(np.mean(list(b0.values()))) if b0 else float('nan')
        g['beta0_fiber'] = b0
        ans['P6'] = g; t['P6'] = time.time() - t0

        if len(explain_inputs):
            t0 = time.time()
            ans['P2'] = p2_explain(ld, fit, explain_inputs, coords=p1['coords'], psi=psi)
            t['P2'] = time.time() - t0
        if holdout is not None and ood:
            t0 = time.time(); ans['P4'] = p4_ignorance(ld, fit, holdout, ood); t['P4'] = time.time() - t0
        if failure_radius is not None:
            t0 = time.time(); ans['P5'] = p5_failure(ld, fit, failure_radius); t['P5'] = time.time() - t0
        if nuisance:
            t0 = time.time()
            ans['P3'] = p3_spurious(ld, fit, nuisance, coords=p1['coords'])
            t['P3'] = time.time() - t0

        rep['answers'][name] = ans
        rep['timings'][name] = t
        if verbose:
            print(f'  [{name}] ' + ' '.join(f'{k2} {v:.1f}s' for k2, v in t.items()), flush=True)
    rep['wall_s'] = time.time() - t00
    return rep
