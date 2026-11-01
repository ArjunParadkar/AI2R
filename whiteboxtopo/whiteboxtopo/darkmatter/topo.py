"""Test 2 -- the v1 topology-with-shuffle-null instrument (TC(H1), beta_1, feature-shuffle
null) applied to error point clouds. Reuses ``whiteboxtopo.signals.memorization.memorization_features``
unchanged (PCA -> 30 dims fitted on the cloud itself, VR persistence via ``whiteboxtopo.core``,
95th-percentile filtration truncation, 5 feature-shuffled copies for the ratio's denominator) --
no persistence code is duplicated here. Ported from whitebox-v2 E6 (``e6lib/topo.py``).
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from ..signals.memorization import memorization_features


def tc_features(cloud: np.ndarray, n_components: int = 30, n_shuffle: int = 5, seed: int = 0) -> Dict:
    return memorization_features(np.asarray(cloud, dtype=np.float64), n_components=n_components,
                                 n_shuffle=n_shuffle, seed=seed)


def tc_ratio_bootstrap(E: np.ndarray, n_points: int = 2000, n_boot: int = 20, n_components: int = 30,
                       seed: int = 0) -> Dict:
    """Point estimate on subsample seed 0 (of size min(n_points, len(E))), CI from
    ``n_boot`` further independent uniform-random subsamples (seeds seed+1 .. seed+n_boot).
    E6 SPEC: n_points=2000, n_boot=20 on the full battery; smaller values are fine for
    quick checks (the ratio is reasonably stable, but the CI width grows as n_points shrinks)."""
    E = np.asarray(E, dtype=np.float64)
    n = len(E); m = min(n_points, n)
    feats = []
    for s in range(n_boot + 1):
        rng = np.random.default_rng(seed + s)
        idx = rng.choice(n, m, replace=False)
        feats.append(tc_features(E[idx], n_components=n_components, seed=seed + s))
    point = feats[0]
    ratios = np.array([f["tc_ratio"] for f in feats[1:]]) if n_boot > 0 else np.array([point["tc_ratio"]])
    b1s = np.array([f["b1"] for f in feats[1:]]) if n_boot > 0 else np.array([point["b1"]])
    return dict(tc_ratio=point["tc_ratio"], tc=point["tc"], tc_shuffled=point["tc_shuffled"], b1=point["b1"],
               pe1=point["pe1"], n_bars=point["n_bars"], med_dist=point["med_dist"],
               tc_ratio_ci=(float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))),
               tc_ratio_boot_mean=float(ratios.mean()), b1_boot_mean=float(b1s.mean()),
               n_points=m, n_boot=n_boot, samples=ratios)
