"""E6 reusable core — the future whitebox.darkmatter.

residual_structure(errors) → dict(structured_fraction, n_outliers, mp_edge, tw_threshold,
                                 excess_fraction, tc_ratio, tc_ratio_ci, b1, shuffle_n_outliers, ...)
"""
from __future__ import annotations
import numpy as np
from .rmt import (mp_outliers, feature_shuffle, gaussian_matched_diag, gaussian_matched_cov, mp_edges,
                  tw_threshold, mp_density, block_bootstrap_outliers, project_out_top_pcs, MPResult)
from .topo import tc_ratio_bootstrap, tc_features


def residual_structure(errors: np.ndarray, *, form: str = "correlation", q: float = 0.99, topology: bool = True,
                       n_points: int = 2000, n_boot: int = 10, n_components: int = 30, seed: int = 0,
                       n_jobs: int = 1) -> dict:
    """Structured-fraction of a residual/error matrix (tokens × dims) under an MP null,
    plus the v1 topology-with-shuffle-null ratio. See E6/SPEC.md for the definitions."""
    E = np.asarray(errors, dtype=np.float64)
    r = mp_outliers(E, form, q)
    rs = mp_outliers(feature_shuffle(E, seed), form, q)
    out = dict(n=r.n, p=r.p, gamma=r.gamma, form=form, mp_edge=r.mp_edge, tw_threshold=r.tw_threshold,
               n_outliers=r.n_outliers, structured_fraction=r.structured_fraction, excess_fraction=r.excess_fraction,
               shuffle_n_outliers=rs.n_outliers, test_valid=bool(rs.n_outliers <= 1), eigenvalues=r.eigenvalues)
    if topology:
        t = tc_ratio_bootstrap(E, n_points=n_points, n_boot=n_boot, n_components=n_components, seed=seed, n_jobs=n_jobs)
        out.update(tc_ratio=t["tc_ratio"], tc_ratio_ci=t["tc_ratio_ci"], ci=t["tc_ratio_ci"], b1=t["b1"],
                   tc=t["tc"], tc_shuffled=t["tc_shuffled"], pe1=t["pe1"])
    return out


__all__ = ["residual_structure", "mp_outliers", "feature_shuffle", "gaussian_matched_diag", "gaussian_matched_cov",
           "mp_edges", "tw_threshold", "mp_density", "block_bootstrap_outliers", "project_out_top_pcs",
           "tc_ratio_bootstrap", "tc_features", "MPResult"]
