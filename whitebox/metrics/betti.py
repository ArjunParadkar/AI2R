"""Betti numbers at a scale, read off a persistence diagram.

beta_k(s) = #{bars with birth <= s < death}. The original code used two
different scale conventions: the median PAIRWISE DISTANCE of the point cloud
(paper / vision runs) and the median BAR MIDPOINT of the diagram (expA GPT-2
run). Both are exposed; ``median_scale`` implements the diagram-only version
because it needs no access to the point cloud.
"""
from __future__ import annotations
import numpy as np


def betti_at_scale(D, scale: float) -> int:
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return 0
    return int(np.sum((D[:, 0] <= scale) & (D[:, 1] > scale)))


def median_scale(D) -> float:
    """Median bar midpoint of a diagram (the expA convention). 0.0 if empty."""
    D = np.asarray(D, dtype=np.float64)
    if D.size == 0:
        return 0.0
    return float(np.median((D[:, 0] + D[:, 1]) / 2.0))


def betti_curve(D, scales) -> np.ndarray:
    scales = np.asarray(scales, dtype=np.float64)
    return np.asarray([betti_at_scale(D, s) for s in scales], dtype=int)
