import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e6lib import mp_outliers, feature_shuffle, residual_structure, gaussian_matched_diag
from e6lib.rmt import mp_edges
from e6lib.probe import RidgeAccumulator, R2Evaluator

N, D = 4000, 200


def test_pure_gaussian_no_outliers():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(N, D)) * rng.uniform(0.5, 3.0, size=D)   # heterogeneous diagonal
    r = mp_outliers(G, "correlation"); assert r.n_outliers <= 1, r.n_outliers
    assert r.structured_fraction < 0.02
    r2 = mp_outliers(rng.normal(size=(N, D)), "covariance"); assert r2.n_outliers <= 1


def _planted(k=5, seed=1):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(N, D))
    gamma = D / N; _, hi = mp_edges(gamma)
    U = np.linalg.qr(rng.normal(size=(D, k)))[0]
    spikes = rng.normal(size=(N, k)) * np.sqrt(5 * hi)     # spike variance 5·λ+ ≫ BBP threshold
    return G + spikes @ U.T


def test_planted_rank_k_recovered():
    P = _planted(5)
    r = mp_outliers(P, "correlation"); assert r.n_outliers == 5, r.n_outliers
    r = mp_outliers(P, "covariance"); assert r.n_outliers == 5, r.n_outliers
    assert r.structured_fraction > 0.05


def test_shuffle_null_kills_planted():
    P = _planted(5)
    r = mp_outliers(feature_shuffle(P, 0), "correlation"); assert r.n_outliers <= 1, r.n_outliers
    g = mp_outliers(gaussian_matched_diag(P, 0), "correlation"); assert g.n_outliers <= 1


def test_residual_structure_api():
    P = _planted(3)
    out = residual_structure(P, n_points=300, n_boot=2)
    for k in ("structured_fraction", "n_outliers", "mp_edge", "tc_ratio", "ci", "test_valid"):
        assert k in out
    assert out["n_outliers"] == 3 and out["test_valid"]
    assert out["ci"][0] <= out["ci"][1]


def test_ridge_probe_recovers_linear_target():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3000, 20)); W = rng.normal(size=(20, 50))
    Y = Z @ W + 0.1 * rng.normal(size=(3000, 50))
    acc = RidgeAccumulator(20, 50, 1e-3); acc.add(torch.as_tensor(Z[:2000]), torch.as_tensor(Y[:2000])); acc.fit()
    ev = R2Evaluator(acc.train_mean()); ev.add(torch.as_tensor(Y[2000:]), acc.predict(torch.as_tensor(Z[2000:])))
    assert ev.r2 > 0.95
    Zs = feature_shuffle(Z[:2000], 0)
    acc2 = RidgeAccumulator(20, 50, 1e-3); acc2.add(torch.as_tensor(Zs), torch.as_tensor(Y[:2000])); acc2.fit()
    ev2 = R2Evaluator(acc2.train_mean()); ev2.add(torch.as_tensor(Y[2000:]), acc2.predict(torch.as_tensor(Z[2000:])))
    assert abs(ev2.r2) < 0.05
