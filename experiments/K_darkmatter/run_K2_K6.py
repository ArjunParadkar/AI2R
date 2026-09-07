"""K2–K5 analysis on the K1 cache (CPU) and K6 certification (SPEC). Writes results/K2_spectral.csv,
K3_topology.csv, K4_predictive.csv, K5_nonlinear.csv, and figures."""
import os, sys, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from whiteboxtopo import nulls
from whiteboxtopo.darkmatter import mp_outliers, block_bootstrap_outliers, gaussian_matched_diag, gaussian_matched_cov, tc_ratio_bootstrap
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "cache"); R = os.path.join(HERE, "results"); FIG = os.path.join(R, "fig")
os.makedirs(FIG, exist_ok=True)
LAYERS = [3, 6, 9]


def k2(L, E, seq):
    r = {}
    for form in ("correlation", "covariance"):
        m = mp_outliers(E, form=form); r[f"n_outliers_{form}"] = m.n_outliers; r[f"structured_fraction_{form}"] = m.structured_fraction
        r[f"tw_threshold_{form}"] = m.tw_threshold; r[f"bulk_edge_{form}"] = m.mp_edge_hi
    bb = block_bootstrap_outliers(E, seq, B=50, seed=L); r.update({f"boot_{k}": v for k, v in bb.items()} if isinstance(bb, dict) else {"boot": str(bb)})
    r["kill_gaussian_matched_marginals"] = mp_outliers(gaussian_matched_diag(E, seed=L)).n_outliers
    r["kill_feature_shuffled"] = mp_outliers(nulls.feature_shuffle(E, seed=L)).n_outliers
    r["control_gaussian_matched_cov"] = mp_outliers(gaussian_matched_cov(E, seed=L)).n_outliers
    ev = np.linalg.eigvalsh(np.corrcoef(E.T))[::-1]
    r["pass_K2"] = bool(r["n_outliers_correlation"] > 10 and r["kill_gaussian_matched_marginals"] <= 1 and r["kill_feature_shuffled"] <= 1)
    return r, ev


def k3(L, E):
    t = tc_ratio_bootstrap(E, n_points=2000, n_boot=20, n_components=20, seed=L)
    G = gaussian_matched_cov(E, seed=L + 7); tg = tc_ratio_bootstrap(G, n_points=2000, n_boot=20, n_components=20, seed=L)
    out = {f"{k}": v for k, v in (t.items() if isinstance(t, dict) else vars(t).items()) if np.isscalar(v) or isinstance(v, (tuple, list))}
    out.update({f"gauss_{k}": v for k, v in (tg.items() if isinstance(tg, dict) else vars(tg).items()) if np.isscalar(v) or isinstance(v, (tuple, list))})
    return out


def ridge_r2(Ztr, Ytr, Zte, Yte, lam=1.0):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8; Ztr = (Ztr - mu) / sd; Zte = (Zte - mu) / sd
    ym = Ytr.mean(0); A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]); W = np.linalg.solve(A, Ztr.T @ (Ytr - ym))
    P = Zte @ W + ym; return float(1 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())


def k4_k5(L, E, D, seq):
    hold = (seq % 5 == 0); val = (seq % 5 == 1)
    tr, te = ~hold & ~val, hold
    D = D.astype(np.float64)
    best = None
    for lam in (0.1, 1.0, 10.0, 100.0):
        r = ridge_r2(E[tr], D[tr], E[val], D[val], lam); best = (r, lam) if best is None or r > best[0] else best
    lam = best[1]
    r2_e = ridge_r2(E[tr], D[tr], E[te], D[te], lam)
    Es = nulls.feature_shuffle(E, seed=L); r2_s = ridge_r2(Es[tr], D[tr], Es[te], D[te], lam)
    rng = np.random.default_rng(L); P = rng.normal(size=(E.shape[1], 2000)) / np.sqrt(E.shape[1])
    phi = lambda Z: np.hstack([Z, (Z @ P) ** 2])
    r2_q = ridge_r2(phi(E[tr]), D[tr], phi(E[te]), D[te], lam)
    return dict(layer=L, lam=lam, r2_linear=r2_e, r2_shuffled=r2_s, delta_r2_vs_shuffle=r2_e - r2_s, pass_K4=bool(r2_e - r2_s > 0.05)), \
           dict(layer=L, lam=lam, r2_linear=r2_e, r2_quadratic=r2_q, delta_r2_nonlinear=r2_q - r2_e, pass_K5=bool(r2_q - r2_e > 0.02))


def main():
    rows2, rows3, rows4, rows5, spectra = [], [], [], [], {}
    for L in LAYERS:
        z = np.load(os.path.join(CACHE, f"L{L}.npz")); E, seq, D = z["E"].astype(np.float64), z["seq"], z["delta"]
        t0 = time.time(); r2, ev = k2(L, E, seq); r2["layer"] = L; rows2.append(r2); spectra[L] = ev; print("K2", r2, f"{time.time()-t0:.0f}s", flush=True)
        r3 = k3(L, E); r3["layer"] = L; rows3.append(r3); print("K3", r3, flush=True)
        r4, r5 = k4_k5(L, E, D, seq); rows4.append(r4); rows5.append(r5); print("K4", r4, "K5", r5, flush=True)
        pd.DataFrame(rows2).to_csv(os.path.join(R, "K2_spectral.csv"), index=False); pd.DataFrame(rows3).to_csv(os.path.join(R, "K3_topology.csv"), index=False)
        pd.DataFrame(rows4).to_csv(os.path.join(R, "K4_predictive.csv"), index=False); pd.DataFrame(rows5).to_csv(os.path.join(R, "K5_nonlinear.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=300)
    for ax, L in zip(axes, LAYERS):
        ev = spectra[L]; r = [x for x in rows2 if x["layer"] == L][0]
        ax.hist(ev, bins=100, color="#4477AA"); ax.axvline(r["tw_threshold_correlation"], color="#EE6677", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_title(f"L{L}: {r['n_outliers_correlation']} outliers"); ax.set_xlabel("eigenvalue (correlation form)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "K2_spectrum.png")); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
