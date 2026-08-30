"""Test 2 — v1 topology instrument (TC(H1), β1, feature-shuffle null) on error clouds.

Reuses whitebox.signals.memorization.memorization_features unchanged (PCA→30,
ripser VR at 95th-percentile truncation, 5 feature-shuffled copies).
"""
from __future__ import annotations
import numpy as np
from whitebox.signals.memorization import memorization_features


def tc_features(cloud: np.ndarray, n_components: int = 30, n_shuffle: int = 5, seed: int = 0) -> dict:
    f = memorization_features(np.asarray(cloud, dtype=np.float64), n_components=n_components,
                              n_shuffle=n_shuffle, seed=seed)
    return f


def _one(args):
    E, idx, seed, n_components = args
    return tc_features(E[idx], n_components=n_components, seed=seed)


def tc_ratio_bootstrap(E: np.ndarray, n_points: int = 2000, n_boot: int = 20, n_components: int = 30,
                       seed: int = 0, n_jobs: int = 1) -> dict:
    """Point estimate on subsample seed 0, CI from n_boot further independent subsamples (seeds 1..n_boot)."""
    E = np.asarray(E, dtype=np.float64)
    n = len(E); m = min(n_points, n)
    jobs = []
    for s in range(n_boot + 1):
        rng = np.random.default_rng(seed + s)
        idx = rng.choice(n, m, replace=False)
        jobs.append((E, idx, seed + s, n_components))
    if n_jobs > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(n_jobs) as pool:
            feats = pool.map(_one, jobs)
    else:
        feats = [_one(j) for j in jobs]
    point = feats[0]
    ratios = np.array([f["tc_ratio"] for f in feats[1:]]) if n_boot > 0 else np.array([point["tc_ratio"]])
    b1s = np.array([f["b1"] for f in feats[1:]]) if n_boot > 0 else np.array([point["b1"]])
    return dict(tc_ratio=point["tc_ratio"], tc=point["tc"], tc_shuffled=point["tc_shuffled"], b1=point["b1"],
                pe1=point["pe1"], n_bars=point["n_bars"], med_dist=point["med_dist"],
                tc_ratio_ci=(float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))),
                tc_ratio_boot_mean=float(ratios.mean()), b1_boot_mean=float(b1s.mean()),
                n_points=m, n_boot=n_boot, samples=ratios)
