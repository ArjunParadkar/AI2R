"""Probe-set subsampling bootstrap: error bars for diagram-derived metrics.

Persistence diagrams are not a resampling-friendly object (duplicate points
create zero-length bars), so we use *subsampling without replacement*
(Politis, Romano & Wolf 1999; Fasy et al. 2014 for persistence): draw
``n_boot`` subsets of size ``subsample`` from the probe, recompute the
diagram and metric on each. For between-epoch quantities (SV) the SAME index
sets are used at both epochs so the difference is representational, not
sampling noise.

Caveat, stated in the returned object: metrics like TC and beta_1 are
*size-dependent*. The subsample statistics estimate the metric at the
subsample size, not at the full probe size. We therefore report the full-probe
point estimate separately and use the subsample distribution only for the
spread (percentile interval width). This is exactly the ``boot_tc_b1``
protocol of the original expC.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence
import numpy as np

from .persistence import Diagrams, compute_diagrams
from .reduce import Reducer


@dataclass
class BootEstimate:
    point: float           # metric on the full probe
    sub_mean: float        # mean over subsamples
    sub_std: float
    ci_low: float          # percentile CI of the subsample distribution
    ci_high: float
    n_boot: int
    subsample: int
    samples: np.ndarray

    def __repr__(self):
        return (f"BootEstimate(point={self.point:.4g}, sub={self.sub_mean:.4g}±{self.sub_std:.2g}, "
                f"CI95=[{self.ci_low:.4g},{self.ci_high:.4g}], n_boot={self.n_boot}, m={self.subsample})")


def subsample_indices(n: int, subsample: int, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = min(subsample, n)
    return np.stack([rng.choice(n, m, replace=False) for _ in range(n_boot)])


def summarize(samples: np.ndarray, point: float, subsample: int, alpha: float = 0.05) -> BootEstimate:
    s = np.asarray(samples, dtype=float)
    return BootEstimate(float(point), float(s.mean()), float(s.std(ddof=1)) if s.size > 1 else 0.0,
                        float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2))),
                        int(s.size), int(subsample), s)


def bootstrap_diagrams(X: np.ndarray, reducer: Optional[Reducer], idx: np.ndarray, **ph_kw) -> list:
    """Diagrams for each row of index sets ``idx`` (n_boot, m)."""
    return [compute_diagrams(X[i], reducer, **ph_kw) for i in idx]


def bootstrap_metric(
    X: np.ndarray, metric: Callable[[np.ndarray], float], reducer: Optional[Reducer] = None,
    k: int = 1, n_boot: int = 20, subsample: int = 150, seed: int = 0, **ph_kw,
) -> BootEstimate:
    """Metric of the dim-k diagram with subsampling error bars."""
    full = compute_diagrams(X, reducer, **ph_kw)
    idx = subsample_indices(len(X), subsample, n_boot, seed)
    vals = [metric(D[k]) for D in bootstrap_diagrams(X, reducer, idx, **ph_kw)]
    return summarize(vals, metric(full[k]), idx.shape[1])
